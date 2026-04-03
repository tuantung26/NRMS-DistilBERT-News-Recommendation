# NRMS-DistilBERT-News-Recommendation
A personalized news recommendation system upgrading the NRMS model with contextual DistilBERT embeddings. Includes offline caching for GPU efficiency and a live Streamlit demo.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Architectural Evolution](#architectural-evolution)
4. [Performance Results](#performance-results)
5. [Streamlit Demo Application](#streamlit-demo-application)
6. [Quick Start](#quick-start)
7. [Documentation & Credits](#documentation--credits)

# NRMS-DistilBERT-News-Recommendation

A next-generation personalized news recommendation system that combines the NRMS framework's powerful multi-head self-attention mechanism with DistilBERT's highly efficient contextual embeddings. Designed for high scalability and real-world applicability, this project includes offline caching for GPU efficiency and a user-friendly Streamlit demo.

**Why this matters:** Traditional news recommendation systems often sacrifice contextual understanding and positional awareness. With this system, we've not only enhanced accuracy but addressed practical deployment challenges.

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
To configure and run the live recommendation demo:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/tuantung26/NRMS-DistilBERT-News-Recommendation.git
cd NRMS-DistilBERT-News-Recommendation
```

2. Install the necessary Python packages:
```bash
pip install -r demo/requirements.txt
```

3. **Download the Offline Caches:** Due to GitHub file limits, download the zipped multi-gigabyte cache binaries from **[Cloud Storage / Kaggle Link Placeholder]** and extract the contents directly into the local `/demo_export/` directory.

4. Launch the Streamlit application:
```bash
streamlit run demo/app.py
```


## Documentation & Credits
* Full Report: Detailed technical analysis and mathematical formulations can be found in the [Technical Paper (PDF)](./paper/main.pdf). located in the docs folder.
* Project Context: Originally developed as a team project at FPT University within other members' contributions.

