# NRMS-DistilBERT-News-Recommendation
A personalized news recommendation system upgrading the NRMS model with contextual DistilBERT embeddings. Includes offline caching for GPU efficiency and a live Streamlit demo.

# Neural News Recommendation with Multi-Head Self-Attention & DistilBERT

[cite_start]This project presents an architectural upgrade of the **NRMS** (Neural News Recommendation with Multi-Head Self-Attention) framework by integrating **DistilBERT** contextual embeddings to replace traditional static GloVe word vectors[cite: 1, 9].

---

## 📌 Project Overview
[cite_start]The system is built and evaluated on the **Microsoft News Dataset (MIND)**, a large-scale benchmark for news recommendation[cite: 18, 58]. [cite_start]The primary goal was to resolve the "positional blindness" of baseline models, enabling the network to understand word order, syntax, and deep semantic context within news titles and abstracts[cite: 8, 24, 27].

* [cite_start]**Contextual Embeddings**: Transitioned from GloVe to DistilBERT to capture dynamic word meanings based on surrounding context[cite: 9, 25].
* [cite_start]**Positional Awareness**: Leveraged Transformer-based positional encodings to provide spatial awareness to the self-attention layers[cite: 11, 84].
* [cite_start]**Offline Caching**: Developed a high-efficiency caching pipeline to handle 230,000+ impression logs within the hardware constraints of Kaggle T4 GPUs[cite: 13, 61, 137].

---

## 🏗️ Architectural Evolution
[cite_start]The project followed a three-stage iterative development process to optimize news and user representations[cite: 28, 168]:

### 1. News Encoder (Section 4.1)
* [cite_start]**Iteration 1 (Baseline)**: Replicated the original NRMS using 300-dimensional static GloVe vectors[cite: 170].
* [cite_start]**Iteration 2 (Frozen DistilBERT)**: Replaced GloVe with a pre-trained DistilBERT feature extractor to inject positional identity[cite: 172, 173].
* [cite_start]**Iteration 3 (Fine-tuned DistilBERT)**: Fine-tuned DistilBERT attention heads specifically on MIND categories to specialize the embeddings for news syntax[cite: 10, 174].



### 2. User Encoder & Click Predictor
* [cite_start]**User Encoder**: Forges a user interest profile $u$ by modeling the conceptual relationships between historically clicked articles using Multi-Head Self-Attention[cite: 20, 95].
* [cite_start]**Click Predictor**: Calculates the engagement probability $\hat{y}$ using a computationally efficient dot product between the user vector and candidate article vector: $\hat{y}=u^{T}r^{c}$[cite: 97].

---

## 📊 Performance Results
[cite_start]Results evaluated on the **MIND-small validation set**[cite: 188]:

| Iteration | Model Configuration | AUC | MRR | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Baseline (Static GloVe) | 0.6174 | 0.3395 | 0.3829 |
| 2 | Frozen Pre-Trained DistilBERT | 0.6618 | 0.3695 | 0.4138 |
| **3** | **Domain Fine-Tuned DistilBERT** | **0.6792** | **0.3800** | **0.4260** |

[cite_start]*The transition to DistilBERT yielded a **7.19% AUC increase** over the baseline[cite: 190].*

---

## 💻 Streamlit Demo Application
[cite_start]To validate the model's real-world scalability, we deployed an interactive demonstration using **Streamlit**[cite: 195, 196]:
* [cite_start]**User Dashboard**: Select from various mock profiles representing specific reading tastes[cite: 198, 199].
* [cite_start]**Live Inference**: The backend loads pre-computed features from the local cache and generates ranked recommendations in real-time[cite: 201, 209].



---

## 🛠️ Quick Start
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/tuantung26/DAP391m_NewS.git](https://github.com/tuantung26/DAP391m_NewS.git)
