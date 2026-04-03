import os, sys, math, html
import numpy as np
import torch
import streamlit as st
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_mappings, load_news, load_behaviors, load_caches, load_entity_embedding, FeatureBuilder
from model import load_model

st.set_page_config(page_title="NRMS Demo", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
* { font-family: 'JetBrains Mono', monospace !important; }
.block-container { max-width: 1200px; }

.news-card {
    border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px;
    margin-bottom: 10px; background: #fff; color: #222;
}
.badge {
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:0.7rem; font-weight:600; margin-right:4px;
}
.cat { background:#e8e8ff; color:#333; }
.sub { background:#f0f0f0; color:#666; }
.clicked { background:#d4edda; color:#155724; }
.not-clicked { background:#f8d7da; color:#721c24; }
.score { float:right; font-weight:700; font-size:0.9rem; }
.score-hi { color:#155724; }
.score-lo { color:#721c24; }
.rank {
    display:inline-block; width:24px; height:24px; background:#333;
    color:#fff; text-align:center; line-height:24px; border-radius:50%;
    font-size:0.75rem; margin-right:8px;
}
.card-title { font-weight:600; margin-top:8px; color:#111; }
.card-abstract { font-size:0.8rem; color:#666; margin-top:4px; }
.card-reason { font-size:0.72rem; color:#888; margin-top:4px; }
.hist-card {
    border:1px solid #eee; border-radius:6px; padding:8px 12px;
    margin-bottom:6px; background:#fff; font-size:0.85rem; color:#222;
}
.hist-title { font-weight:500; margin-top:4px; color:#111; }
.metric-card {
    border:1px solid #ddd; border-radius:8px; padding:14px;
    text-align:center; background:#fff;
}
.metric-val { font-size:1.6rem; font-weight:700; color:#222; }
.metric-lbl { font-size:0.7rem; color:#888; margin-top:2px; }
.m-green { color:#155724 !important; }
.m-yellow { color:#856404 !important; }
.m-red { color:#721c24 !important; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_all():
    m = load_mappings()
    news_df, lookup = load_news()
    beh = load_behaviors()
    tc, ac = load_caches()
    ee = load_entity_embedding()
    fb = FeatureBuilder(m, tc, ac, lookup)
    model = load_model(m, ee, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo_export", "best_model.pt"))
    return m, news_df, lookup, beh, fb, model

with st.spinner("Loading model and data..."):
    mappings, news_df, news_lookup, behaviors, fb, model = load_all()

def esc(text):
    return html.escape(str(text)) if text else ""

st.markdown("# NRMS News Recommender")
st.caption("Neural News Recommendation with Multi-Head Self-Attention")

with st.sidebar:
    st.markdown("### Settings")
    valid = behaviors[behaviors['History'].str.len() > 0].reset_index(drop=True)
    st.caption(f"{len(valid)} impressions available")

    idx = st.number_input("Impression", 0, len(valid)-1, 0, 1)
    row = valid.iloc[idx]
    uid = row['UserID']
    st.text(f"User: {uid}")
    st.text(f"Time: {row.get('Time', '')}")

    history = [n for n in str(row['History']).split() if n in news_lookup]
    imps_raw = str(row['Impressions']).split()
    st.text(f"History: {len(history)} articles")
    st.text(f"Candidates: {len(imps_raw)} articles")

    st.divider()
    max_cand = st.slider("Max candidates", 5, min(100, len(imps_raw)), min(30, len(imps_raw)), 5)
    st.divider()
    go = st.button("Get Recommendations", type="primary", use_container_width=True)

candidates = []
for imp in imps_raw[:max_cand]:
    if '-' in imp:
        nid, lbl = imp.rsplit('-', 1)
        candidates.append((nid, int(lbl)))
    else:
        candidates.append((imp, -1))

c1, c2, c3, c4 = st.columns(4)
clicked_count = sum(1 for _, l in candidates if l == 1)
for col, val, lbl in [(c1, len(history), "History"), (c2, len(candidates), "Candidates"),
                       (c3, clicked_count, "Clicked"), (c4, uid[:8], "User")]:
    col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("")
left, right = st.columns([1, 2])

with left:
    st.markdown("### Reading History")
    st.caption("Turn off articles or add new ones to see how recommendations change")

    # Interactive history selection
    active_history = []
    
    # State for manually added history items
    state_key = f"added_hist_{uid}"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
        
    # Search and add new articles
    all_nids = list(news_lookup.keys())
    # Create a formatted display string for the dropdown
    def format_news(nid):
        n = news_lookup[nid]
        cat = n.get("Category", "")
        title = n.get("Title", nid)
        if len(title) > 60: title = title[:60] + "..."
        return f"[{cat}] {title}"
        
    selected_to_add = st.multiselect(
        "Search & Add to History", 
        options=all_nids,
        format_func=format_news,
        placeholder="Type to search articles..."
    )
    
    if selected_to_add:
        for nid in selected_to_add:
            if nid not in st.session_state[state_key] and nid not in history:
                st.session_state[state_key].append(nid)
    
    def remove_added(n_id):
        if n_id in st.session_state[state_key]:
            st.session_state[state_key].remove(n_id)

    # Combined history: manually added (at top/most recent) + original
    combined_history = st.session_state[state_key] + history
    
    st.markdown("<br>", unsafe_allow_html=True) # Prevent text overlap from multiselect pills
    
    # Show up to 20 history items
    for idx, nid in enumerate(combined_history[:20]):
        if nid in news_lookup:
            n = news_lookup[nid]
            is_added = nid in st.session_state[state_key]
            
            # Use session state to remember checkbox values
            key = f"hist_{uid}_{idx}_{nid}"
            
            # Create a custom row
            col1, col2 = st.columns([1, 15] if not is_added else [2, 14])
            
            with col1:
                if is_added:
                    st.button("🗑️", key=f"del_{key}", help="Remove manually added article", on_click=remove_added, args=(nid,))
                    is_active = True # Manually added articles are always active unless removed
                else:
                    is_active = st.checkbox("", value=True, key=key, label_visibility="collapsed")
            
            with col2:
                opacity = "1.0" if is_active else "0.4"
                bg_color = "#fffbeb" if is_added else "#fff" # Light yellow for added items
                border = "1px solid #ffeeba" if is_added else "1px solid #eee"
                
                is_added_badge = '<span class="badge" style="background:#ffc107;color:#000;">ADDED</span>' if is_added else ''
                
                st.markdown(
                    f'<div class="hist-card" style="opacity: {opacity}; background: {bg_color}; border: {border};">'
                    f'{is_added_badge}'
                    f'<span class="badge cat">{esc(n.get("Category",""))}</span>'
                    f'<span class="badge sub">{esc(n.get("SubCategory",""))}</span>'
                    f'<div class="hist-title">{esc(n.get("Title", nid))}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            if is_active:
                active_history.append(nid)

    if len(combined_history) > 20:
        st.caption(f"...and {len(combined_history) - 20} more older articles (always active)")
        active_history.extend(combined_history[20:])

    st.markdown("---")
    
    # Category breakdown based on ACTIVE history
    cats = [news_lookup[n].get("Category","?") for n in active_history if n in news_lookup]
    if cats:
        counts = Counter(cats)
        st.caption(f"Category interests (based on {len(active_history)} active articles)")
        for cat, cnt in counts.most_common(8):
            pct = cnt / len(cats) * 100
            st.progress(pct / 100, text=f"{cat} ({cnt}, {pct:.0f}%)")

with right:
    # Use active_history instead of full history for inference
    if go:
        if len(active_history) == 0:
            st.warning("You disabled all history articles! The model will make predictions based on a completely empty user profile (cold start).")
        st.markdown("### Recommendations")

        cand_nids = [nid for nid, _ in candidates]
        cand_labels = {nid: l for nid, l in candidates}

        with st.spinner("Running inference..."):
            batch = fb.build_batch(active_history, cand_nids)
            with torch.no_grad():
                scores = model(batch).squeeze(0).cpu().numpy()

        ranked = sorted(zip(cand_nids, scores), key=lambda x: -x[1])

        user_cats = Counter(news_lookup[n].get("Category","") for n in active_history if n in news_lookup)

        for rank, (nid, score) in enumerate(ranked, 1):
            if nid not in news_lookup: continue
            n = news_lookup[nid]
            lbl = cand_labels.get(nid, -1)

            click_html = ""
            if lbl == 1: click_html = '<span class="badge clicked">CLICKED</span>'
            elif lbl == 0: click_html = '<span class="badge not-clicked">NOT CLICKED</span>'

            sc_class = "score-hi" if score > 0 else "score-lo"

            abstract = esc(n.get("Abstract", ""))
            if len(abstract) > 180: abstract = abstract[:180] + "..."

            article_cat = n.get("Category", "")
            cat_match = user_cats.get(article_cat, 0)
            cat_total = sum(user_cats.values())
            match_pct = cat_match / cat_total * 100 if cat_total > 0 else 0

            reason_html = ""
            if match_pct > 20:
                reason_html = f'<div class="card-reason">Category match: {esc(article_cat)} appears in {match_pct:.0f}% of history</div>'
            elif match_pct == 0:
                reason_html = f'<div class="card-reason">New category for this user ({esc(article_cat)})</div>'

            st.markdown(
                f'<div class="news-card">'
                f'<span class="rank">{rank}</span>'
                f'<span class="{sc_class} score">{score:.4f}</span>'
                f'<span class="badge cat">{esc(article_cat)}</span>'
                f'<span class="badge sub">{esc(n.get("SubCategory",""))}</span>'
                f'{click_html}'
                f'<div class="card-title">{esc(n.get("Title", nid))}</div>'
                f'<div class="card-abstract">{abstract}</div>'
                f'{reason_html}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("### Model Performance")

        labels = [1 if cand_labels.get(nid)==1 else 0 for nid, _ in ranked]

        mrr = 0.0
        for i, l in enumerate(labels):
            if l == 1: mrr = 1/(i+1); break

        def ndcg(labels, k):
            dcg = sum(labels[i]/math.log2(i+2) for i in range(min(k,len(labels))))
            ideal = sorted(labels, reverse=True)
            idcg = sum(ideal[i]/math.log2(i+2) for i in range(min(k,len(ideal))))
            return dcg/idcg if idcg>0 else 0

        def auc(labels, scores_list):
            pos = [s for l,s in zip(labels, scores_list) if l==1]
            neg = [s for l,s in zip(labels, scores_list) if l==0]
            if not pos or not neg: return 0.5
            return sum(1 for p in pos for n in neg if p>n) / (len(pos)*len(neg))

        n5 = ndcg(labels, 5)
        n10 = ndcg(labels, 10)
        a = auc(labels, [s for _, s in ranked])

        def mcolor(v):
            if v >= 0.6: return "m-green"
            if v >= 0.3: return "m-yellow"
            return "m-red"

        mc = st.columns(4)
        for col, name, val, tip in [
            (mc[0], "MRR", mrr, "How quickly the first clicked article is found"),
            (mc[1], "nDCG@5", n5, "Quality of top-5 ranking"),
            (mc[2], "nDCG@10", n10, "Quality of top-10 ranking"),
            (mc[3], "AUC", a, "Ability to separate clicked vs non-clicked"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-lbl">{name}</div>'
                f'<div class="metric-val {mcolor(val)}">{val:.3f}</div>'
                f'<div class="metric-lbl">{tip}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        clicked_ranks = [r for r,(nid,_) in enumerate(ranked,1) if cand_labels.get(nid)==1]
        if clicked_ranks:
            best = min(clicked_ranks)
            total = len(ranked)
            quality = "top spot" if best==1 else f"top {best/total*100:.0f}%" if best<=total*0.3 else f"position {best}/{total}"
            st.caption(f"Clicked article(s) at position(s) {clicked_ranks} — first appeared at {quality}")
        else:
            st.caption("No clicked articles in this candidate set")
    else:
        st.markdown("### Recommendations")
        st.info("Select a user and click Get Recommendations")
        st.markdown("""
        **How it works:**
        1. Model reads the user's reading history
        2. Scores each candidate article by relevance
        3. Ranks articles — higher score = more relevant
        4. CLICKED/NOT CLICKED shows ground truth

        **Metrics shown after ranking:**
        - MRR — did the model rank a clicked article first?
        - nDCG@5/10 — are clicked articles near the top?
        - AUC — can the model tell clicked from non-clicked?
        """)
