# NRMS-DistilBERT-News-Recommendation
A personalized news recommendation system upgrading the NRMS model with contextual DistilBERT embeddings. Includes offline caching for GPU efficiency and a live Streamlit demo.

# Neural News Recommendation with Multi-Head Self-Attention & DistilBERT

This project presents an architectural upgrade of the NRMS (Neural News Recommendation with Multi-Head Self-Attention) framework by integrating DistilBERT contextual embeddings to replace traditional static GloVe word vectors.

---

## Project Overview
The system is built and evaluated on the Microsoft News Dataset (MIND), a large-scale benchmark for news recommendation. The primary goal was to resolve the "positional blindness" of baseline models, enabling the network to understand word order, syntax, and deep semantic context within news titles and abstracts.

* Contextual Embeddings: Transitioned from GloVe to DistilBERT to capture dynamic word meanings based on surrounding context.
* Positional Awareness: Leveraged Transformer-based positional encodings to provide spatial awareness to the self-attention layers.
* Offline Caching: Developed a high-efficiency caching pipeline to handle 230,000+ impression logs within the hardware constraints of Kaggle T4 GPUs.

---

## Architectural Evolution
The project followed a three-stage iterative development process to optimize news and user representations:

### 1. News Encoder
* Iteration 1 (Baseline): Replicated the original NRMS using 300-dimensional static GloVe vectors.
* Iteration 2 (Frozen DistilBERT): Replaced GloVe with a pre-trained DistilBERT feature extractor to inject positional identity.
* Iteration 3 (Fine-tuned DistilBERT): Fine-tuned DistilBERT attention heads specifically on MIND categories to specialize the embeddings for news syntax.

### 2. User Encoder & Click Predictor
* User Encoder: Forges a user interest profile u by modeling the conceptual relationships between historically clicked articles using Multi-Head Self-Attention.
* Click Predictor: Calculates the engagement probability using a computationally efficient dot product between the user vector and candidate article vector: y = uT * rc.

---

## Performance Results
Results evaluated on the MIND-small validation set:

| Iteration | Model Configuration | AUC | MRR | nDCG@10 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Baseline (Static GloVe) | 0.6174 | 0.3395 | 0.3829 |
| 2 | Frozen Pre-Trained DistilBERT | 0.6618 | 0.3695 | 0.4138 |
| 3 | Domain Fine-Tuned DistilBERT | 0.6792 | 0.3800 | 0.4260 |

The transition to DistilBERT yielded a 7.19% AUC increase over the baseline.

---

## Streamlit Demo Application
To validate the model's real-world scalability, we deployed an interactive demonstration using Streamlit:
* User Dashboard: Select from various mock profiles representing specific reading tastes.
* Live Inference: The backend loads pre-computed features from the local cache and generates ranked recommendations in real-time.

---

## Quick Start
1. Clone the repository:
   git clone https://github.com/tuantung26/DAP391m_NewS.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Dashboard:
   streamlit run streamlit_app/app.py

---

## Documentation & Credits
* Full Report: Detailed technical analysis and mathematical formulations can be found in the Technical Paper (PDF) located in the docs folder.
* Project Context: Originally developed as a team project at FPT University.
* My Contributions: I was primarily responsible for implementing the DistilBERT News Encoder (Iteration 3), architecting the Offline Caching pipeline, and developing the Streamlit application logic.
