# GeoHyFormer: Geometric Hybrid Transformer for Fake News Detection

GeoHyFormer is a **geometric hybrid transformer model** designed for **fake news detection on social media**, integrating both **geometric representation** and **temporal context** of information propagation.  
It builds on graph-based message passing and hierarchical time encoding to effectively capture the dynamic behavior of fake news diffusion.

---

## Project Structure
GeoHyFormer/
│
├── data/
│   ├── gossipcop/
│   │   ├── processed/
│   │   └── raw/
│   └── politifact/
│       ├── processed/
│       └── raw/
│
└── main/
    ├── __init__.py
    ├── evaluate.py
    ├── load_data.py
    ├── model.py
    ├── train.py
    └── utils.py

---

## How to Run

### **Training**
```bash
python main/train.py --name politifact(or gossipcop) --root ./data --epochs 50 --batch_size 32

Evaluation:
python main/evaluate.py --name politifact --root ./data --ckpt ./data/politifact(or gossipcop)/processed/patgt_best.pt

Datasets

We use benchmark datasets from the FakeNewsNet framework:

Politifact

GossipCop

Each dataset is organized into:

raw/ — original post and user data

processed/ — graph-structured and preprocessed data used for training

Link data: https://drive.google.com/drive/u/0/folders/1FqpR-toGrend7P0280OixS2oNV2oqbjK

Author

Developed by Thanh Duong Nhat
GitHub: thanh1374



