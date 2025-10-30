# 🌐 GeoHyFormer: Geometric Hybrid Transformer for Fake News Detection

> **GeoHyFormer** is a geometric hybrid transformer model for **fake news detection on social media**, integrating both **geometric representation** and **temporal context** of information propagation.  
> The model leverages **graph-based message passing** and **multi-scale time encoding** to capture the dynamic diffusion behavior of misinformation.

---

## 🧩 Project Structure

```yaml
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
    ├── evaluate.py
    ├── load_data.py
    ├── model.py
    ├── train.py
    └── utils.py


## 🚀 How to Run

### 🏋️‍♂️ Train the Model

```bash
python main/train.py --name politifact --root ./data --epochs 50 --batch_size 8
🧪 Evaluate
bash
Sao chép mã
python main/evaluate.py --name politifact --root ./data --ckpt ./data/politifact/processed/patgt_best.pt
📊 Results
Dataset	Accuracy (%)	F1-score (%)
Politifact	84.62	84.65
GossipCop	97.23	97.22

GeoHyFormer consistently outperforms baseline models on both datasets.

🧠 Model Overview
GeoHyFormer introduces a Dual-Hybrid Layer that fuses:

🧩 Geometric message passing for structural dependencies

⏱️ Temporal hybrid attention for multi-scale time encoding

Architecture pipeline:

pgsql
Sao chép mã
Input Graph → Time Encoding → DualHybridStack → Multi-view Pooling → Classification
<p align="center"> <img src="assets/architecture.png" alt="GeoHyFormer Architecture" width="600"> </p>
📁 Datasets
We adopt datasets from the UPFD (User Preference-aware Fake News Detection) benchmark:

Politifact 🗳️

GossipCop 🎬

Each dataset includes:

raw/ — original post and user engagement data

processed/ — graph-structured data ready for training

📦 Download link: UPFD Dataset (Google Drive)

⚙️ Requirements
bash
Sao chép mã
pip install torch torch-geometric torch-scatter
pip install numpy pandas tqdm
📚 Reference
csharp
Sao chép mã
Federico Monti, Fabrizio Frasca, Davide Eynard, Damon Mannion, and Michael M. Bronstein.  
Fake news detection on social media using geometric deep learning. ICLR Workshop (2019).
👨‍💻 Author
Developed by: Thanh Duong Nhat
🔗 GitHub: thanh1374

