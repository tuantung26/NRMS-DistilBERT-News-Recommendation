import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, model_dim, dropout=0.0):
        super().__init__()
        self.head_num = head_num
        self.dim_per_head = model_dim // head_num
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, L, D = q.size()
        h, d = self.head_num, self.dim_per_head
        q = q.view(B,L,h,d).transpose(1,2)
        k = k.view(B,L,h,d).transpose(1,2)
        v = v.view(B,L,h,d).transpose(1,2)
        s = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d)
        if mask is not None: s = s.masked_fill(mask.unsqueeze(1)==0, -1e4)
        a = self.dropout(F.softmax(s, dim=-1))
        return torch.matmul(a, v).transpose(1,2).contiguous().view(B,L,D)


class TextEncoder(nn.Module):
    def __init__(self, dim, heads, attn_dim):
        super().__init__()
        self.self_attn_size = dim // heads * heads
        self.weight_query = nn.Parameter(torch.FloatTensor(dim, self.self_attn_size))
        self.weight_key   = nn.Parameter(torch.FloatTensor(dim, self.self_attn_size))
        self.weight_value = nn.Parameter(torch.FloatTensor(dim, self.self_attn_size))
        self.trans_weight_v = nn.Parameter(torch.FloatTensor(self.self_attn_size, attn_dim))
        self.trans_weight_q = nn.Parameter(torch.FloatTensor(attn_dim, 1))
        self.self_attn = MultiHeadedAttention(heads, dim)
        for p in [self.weight_query, self.weight_key, self.weight_value, self.trans_weight_v, self.trans_weight_q]:
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x, sa_mask, mask):
        h = self.self_attn(x@self.weight_query, x@self.weight_key, x@self.weight_value, sa_mask)
        s = (torch.tanh(h @ self.trans_weight_v) @ self.trans_weight_q).squeeze(-1)
        s = s.masked_fill(mask==0, -1e4)
        a = F.softmax(s, dim=1)
        a = a.masked_fill(mask.sum(1, keepdim=True)==0, 0)
        return torch.bmm(h.transpose(1,2), a.unsqueeze(-1)).squeeze(-1)


class NewsEncoder(nn.Module):
    def __init__(self, n_cat, n_sub, entity_embed, cfg, bert_hidden):
        super().__init__()
        E = cfg['news_final_embed_size']
        self.cat_embed = nn.Embedding(n_cat+1, E)
        self.subcat_embed = nn.Embedding(n_sub+1, E)
        self.entity_embed = nn.Embedding.from_pretrained(torch.FloatTensor(entity_embed), freeze=True)

        tp = bert_hidden // cfg['num_head_text'] * cfg['num_head_text']
        ap = bert_hidden // cfg['num_head_text'] * cfg['num_head_text']
        es = entity_embed.shape[1] // cfg['num_head_entity'] * cfg['num_head_entity']

        self.title_proj = nn.Linear(bert_hidden, tp)
        self.abs_proj = nn.Linear(bert_hidden, ap)
        self.title_encoder = TextEncoder(tp, cfg['num_head_text'], cfg['text_attn_vector_size'])
        self.abstract_encoder = TextEncoder(ap, cfg['num_head_text'], cfg['text_attn_vector_size'])
        self.title_entity_encoder = TextEncoder(entity_embed.shape[1], cfg['num_head_entity'], cfg['entity_attn_vector_size'])
        self.abstract_entity_encoder = TextEncoder(entity_embed.shape[1], cfg['num_head_entity'], cfg['entity_attn_vector_size'])

        self.title_linear = nn.Linear(tp, E)
        self.abstract_linear = nn.Linear(ap, E)
        self.title_entity_linear = nn.Linear(es, E)
        self.abstract_entity_linear = nn.Linear(es, E)

        self.trans_weight_v = nn.Parameter(torch.FloatTensor(E, cfg['news_final_attn_vector_size']))
        self.trans_weight_q = nn.Parameter(torch.FloatTensor(cfg['news_final_attn_vector_size'], 1))
        nn.init.uniform_(self.trans_weight_v, -0.1, 0.1)
        nn.init.uniform_(self.trans_weight_q, -0.1, 0.1)

    def forward(self, cat, sub, t_tok, t_msk, a_tok, a_msk, t_ent, t_em, a_ent, a_em):
        c = self.cat_embed(cat).unsqueeze(1)
        s = self.subcat_embed(sub).unsqueeze(1)

        tx = self.title_proj(t_tok.float())
        t = self.title_linear(self.title_encoder(tx, t_msk.unsqueeze(1)*t_msk.unsqueeze(2), t_msk)).unsqueeze(1)
        ax = self.abs_proj(a_tok.float())
        a = self.abstract_linear(self.abstract_encoder(ax, a_msk.unsqueeze(1)*a_msk.unsqueeze(2), a_msk)).unsqueeze(1)

        te = self.entity_embed(t_ent[:,:,0].long())
        te = self.title_entity_linear(self.title_entity_encoder(te, t_em.unsqueeze(1)*t_em.unsqueeze(2), t_em)).unsqueeze(1)
        ae = self.entity_embed(a_ent[:,:,0].long())
        ae = self.abstract_entity_linear(self.abstract_entity_encoder(ae, a_em.unsqueeze(1)*a_em.unsqueeze(2), a_em)).unsqueeze(1)

        x = torch.cat([c, s, t, a, te, ae], dim=1)
        w = F.softmax(torch.tanh(x @ self.trans_weight_v) @ self.trans_weight_q, dim=1)
        return torch.bmm(x.transpose(1,2), w).squeeze(-1)


class NRMS(nn.Module):
    def __init__(self, news_encoder, cfg):
        super().__init__()
        self.news_encoder = news_encoder
        self.history_encoder = TextEncoder(cfg['news_final_embed_size'], cfg['num_head_text'], cfg['his_final_attn_vector_size'])
        self.user_proj = nn.Linear(cfg['news_final_embed_size'], cfg['user_embed_size'])

    def _encode(self, b, prefix):
        keys = ["cat","subcat","title_tok","title_mask","abs_tok","abs_mask",
                "title_ent","title_ent_mask","abs_ent","abs_ent_mask"]
        tensors = [b[f"{prefix}_{k}"] for k in keys]
        B, N = tensors[0].shape
        flat = [t.view(B*N, *t.shape[2:]) if t.dim()>2 else t.view(B*N) for t in tensors]
        return self.news_encoder(*flat).view(B, N, -1)

    def forward(self, batch):
        dev = next(self.parameters()).device
        batch = {k: v.to(dev) for k, v in batch.items()}

        hv = self._encode(batch, "h")
        hm = (batch["h_cat"]!=0).float()
        uv = self.user_proj(self.history_encoder(hv, hm.unsqueeze(1)*hm.unsqueeze(2), hm))

        cv = self.user_proj(self._encode(batch, "c"))
        return torch.bmm(cv, uv.unsqueeze(-1)).squeeze(-1)


def load_model(mappings, entity_embed, path, device="cpu"):
    cfg = mappings["CFG"]
    ne = NewsEncoder(len(mappings["cat2id"]), len(mappings["subcat2id"]),
                     entity_embed, cfg, mappings["bert_hidden"])
    model = NRMS(ne, cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
