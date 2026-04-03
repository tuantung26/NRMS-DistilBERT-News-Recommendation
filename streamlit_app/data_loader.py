import os, json, pickle
import numpy as np
import pandas as pd
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo_export")


def load_mappings():
    with open(os.path.join(DATA_DIR, "mappings.pkl"), "rb") as f:
        return pickle.load(f)


def load_news():
    cols = ['NewsID','Category','SubCategory','Title','Abstract','URL','TitleEntities','AbstractEntities']
    df = pd.read_csv(os.path.join(DATA_DIR, "news.tsv"), sep='\t', header=None, names=cols, dtype=str).fillna("")
    lookup = {row["NewsID"]: row for _, row in df.iterrows()}
    return df, lookup


def load_behaviors():
    cols = ['ImpressionID','UserID','Time','History','Impressions']
    df = pd.read_csv(os.path.join(DATA_DIR, "behaviors_dev.tsv"), sep='\t', header=None, names=cols, dtype=str).fillna("")
    return df


def load_caches():
    title = torch.load(os.path.join(DATA_DIR, "title_cache.pt"), map_location="cpu")
    abstract = torch.load(os.path.join(DATA_DIR, "abstract_cache.pt"), map_location="cpu")
    return title, abstract


def load_entity_embedding():
    vectors = []
    with open(os.path.join(DATA_DIR, "entity_embedding.vec"), 'r') as f:
        first = f.readline().strip().split()
        if len(first) == 2 and all(t.isdigit() for t in first):
            dim = int(first[1])
        else:
            f.seek(0)
            dim = None
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            try:
                float(parts[0]); start = 0
            except ValueError:
                start = 1
            vec = list(map(float, parts[start:]))
            if dim is None: dim = len(vec)
            if len(vec) == dim: vectors.append(vec)
    embed = np.zeros((len(vectors) + 1, dim), dtype=np.float32)
    embed[1:] = np.array(vectors, dtype=np.float32)
    return embed


def parse_entities(json_str, max_ent=10):
    try: arr = json.loads(json_str)
    except: arr = []
    ids = []
    for e in arr:
        if isinstance(e, dict):
            for k in ["EntityId","entity_id","entityId","Index","index"]:
                if k in e and isinstance(e[k], (int, float)):
                    ids.append(int(e[k])); break
    ids = (ids[:max_ent] + [0]*max_ent)[:max_ent]
    return np.array(ids, dtype=np.int64).reshape(max_ent, 1)


class FeatureBuilder:
    def __init__(self, mappings, title_cache, abs_cache, news_lookup):
        self.m = mappings
        self.tc = title_cache
        self.ac = abs_cache
        self.news = news_lookup
        self.nid2idx = mappings["nid2idx"]
        self.cfg = mappings["CFG"]
        self.H = mappings["bert_hidden"]
        self.E = mappings["MAX_ENTITY"]
        self.fp16 = self.cfg.get("use_fp16_cache", True)

    def _bert(self, nid, which):
        if nid not in self.nid2idx:
            L = self.cfg["max_title_len"] if which == "title" else self.cfg["max_abs_len"]
            return torch.zeros(L, self.H, dtype=torch.float16 if self.fp16 else torch.float32), torch.zeros(L)
        i = self.nid2idx[nid]
        cache = self.tc if which == "title" else self.ac
        return cache["token_embs"][i], cache["attn_masks"][i]

    def feat(self, nid):
        if nid == "PAD" or nid not in self.news:
            t_tok, t_msk = self._bert("__PAD__", "title")
            a_tok, a_msk = self._bert("__PAD__", "abs")
            return dict(cat=0, subcat=0, title_tok=t_tok, title_mask=t_msk,
                        abs_tok=a_tok, abs_mask=a_msk,
                        title_ent=torch.zeros(self.E,1,dtype=torch.long), title_ent_mask=torch.zeros(self.E),
                        abs_ent=torch.zeros(self.E,1,dtype=torch.long), abs_ent_mask=torch.zeros(self.E))
        r = self.news[nid]
        t_tok, t_msk = self._bert(nid, "title")
        a_tok, a_msk = self._bert(nid, "abs")
        t_ent = torch.from_numpy(parse_entities(r["TitleEntities"], self.E))
        a_ent = torch.from_numpy(parse_entities(r["AbstractEntities"], self.E))
        return dict(
            cat=self.m["cat2id"].get(str(r["Category"]), 0),
            subcat=self.m["subcat2id"].get(str(r["SubCategory"]), 0),
            title_tok=t_tok, title_mask=t_msk.float(),
            abs_tok=a_tok, abs_mask=a_msk.float(),
            title_ent=t_ent.long(), title_ent_mask=(t_ent[:,0]!=0).float(),
            abs_ent=a_ent.long(), abs_ent_mask=(a_ent[:,0]!=0).float())

    def build_batch(self, history_nids, candidate_nids):
        max_his = self.m["MAX_HIS"]
        history = (history_nids[:max_his] + ["PAD"] * max_his)[:max_his]
        h = [self.feat(n) for n in history]
        c = [self.feat(n) for n in candidate_nids]

        def stack(feats, p):
            return {
                f"{p}_cat": torch.tensor([f["cat"] for f in feats]).unsqueeze(0),
                f"{p}_subcat": torch.tensor([f["subcat"] for f in feats]).unsqueeze(0),
                f"{p}_title_tok": torch.stack([f["title_tok"] for f in feats]).unsqueeze(0),
                f"{p}_title_mask": torch.stack([f["title_mask"] for f in feats]).unsqueeze(0),
                f"{p}_abs_tok": torch.stack([f["abs_tok"] for f in feats]).unsqueeze(0),
                f"{p}_abs_mask": torch.stack([f["abs_mask"] for f in feats]).unsqueeze(0),
                f"{p}_title_ent": torch.stack([f["title_ent"] for f in feats]).unsqueeze(0),
                f"{p}_title_ent_mask": torch.stack([f["title_ent_mask"] for f in feats]).unsqueeze(0),
                f"{p}_abs_ent": torch.stack([f["abs_ent"] for f in feats]).unsqueeze(0),
                f"{p}_abs_ent_mask": torch.stack([f["abs_ent_mask"] for f in feats]).unsqueeze(0),
            }
        return {**stack(h, "h"), **stack(c, "c")}
