# GeoHyFormer: Geometric Hybrid Transformer for Fake News Detection

> **GeoHyFormer** is a geometric hybrid transformer model for **fake news detection on social media**, integrating both **geometric representation** and **temporal context** of information propagation.  
> The model leverages **graph-based message passing** and **multi-scale time encoding** to capture the dynamic diffusion behavior of misinformation.

---

## Project Structure

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
```
##  How to Run

### Train the Model

```bash
python main/train.py --name politifact (or gossipcop) --root ./data --epochs 50 --batch_size 32
```
### Evaluate
```bash
python main/evaluate.py --name politifact (or gossipcop) --root ./data --ckpt ./data/politifact/processed/patgt_best.pt
```
Results
| Dataset | Accuracy (%) | F1-score (%) |
|-------|-------|-------|
| Politifact | 89.14 | 89.13 |
| GossipCop | 97.67 | 97.67 |

GeoHyFormer consistently outperforms baseline models on both datasets.
### Datasets
We adopt datasets from the FakeNewsNet benchmark:

Politifact

GossipCop

Each dataset includes:

raw/ — original post and user engagement data

processed/ — graph-structured data ready for training

Download link (Google Drive):
```bash
https://drive.google.com/drive/u/0/folders/1FqpR-toGrend7P0280OixS2oNV2oqbjK
```
Developed by: Thanh Duong Nhat
🔗 GitHub: thanh1374

