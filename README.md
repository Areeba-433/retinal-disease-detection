# 🔬 Retinal Disease Detection using Deep Learning

> Multi-label eye disease classifier detecting **8 diseases simultaneously** from fundus retinal images  
> Built by **Areeba Mumtaz Minhas** · PUCIT, University of the Punjab · Class of 2027

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 🎯 Problem Statement

Over **450 million people** worldwide live with diabetes. Diabetic retinopathy — damage to the retina caused by diabetes — is the **leading cause of blindness in working-age adults**, yet it is almost entirely preventable with early detection.

This project builds an AI system that can screen a retinal fundus photograph and simultaneously flag 8 different eye conditions, enabling faster, scalable diagnosis — especially in regions where ophthalmologists are scarce.

---

## 📊 Results

Trained for **10 epochs on CPU** (no GPU required):

| Disease | AUC Score |
|---------|-----------|
| 👁️ Myopia | **0.9826** |
| 👁️ Cataract | **0.9222** |
| 👁️ Glaucoma | **0.8760** |
| 👁️ AMD | **0.8739** |
| 👁️ Hypertension | **0.8223** |
| 👁️ Diabetes | **0.7992** |
| 👁️ Normal | **0.7787** |
| 👁️ Other | **0.7026** |
| 📈 **Mean AUC** | **0.8452** |

> AUC-ROC score: 1.0 = perfect, 0.5 = random. Scores above 0.85 are considered clinically strong.

---

## 🏗️ Architecture

```
Raw Fundus Image (JPG)
        │
        ▼
CLAHE Contrast Enhancement (OpenCV)
        │
        ▼
Albumentations Augmentation
(Flip · Rotate · ColorJitter)
        │
        ▼
EfficientNet-B0 Backbone
(ImageNet pretrained → fine-tuned)
1280-dim feature embedding
        │
        ▼
Multi-label Classification Head
Dropout(0.3) → Linear(512) → ReLU → Linear(8)
        │
        ▼
8 Disease Probabilities
BCE with Logits Loss
```

**Key design decisions:**

- **EfficientNet-B0** chosen for best accuracy/speed tradeoff on CPU
- **CLAHE preprocessing** dramatically improves vessel and lesion visibility in retinal images
- **Multi-label BCE loss** — one image can have multiple diseases simultaneously
- **Per-disease AUC-ROC** — the standard metric used in clinical literature

---

## 🗂️ Project Structure

```
retinal-disease-detection/
├── src/
│   ├── dataset.py       # Data loading + CLAHE preprocessing + augmentation
│   ├── model.py         # EfficientNet-B0 backbone + multi-label classifier head
│   ├── train.py         # Training loop + AUC-ROC evaluation per disease
│   ├── evaluate.py      # GradCAM heatmap generation + explainability
│   ├── app.py           # Gradio web demo (local + public shareable link)
│   └── utils.py         # ROC curves · loss plots · AUC bar charts · CSV logging
├── notebooks/
│   └── explore.ipynb    # Data exploration notebook
├── data/                # Dataset (not included — see Setup below)
├── outputs/             # Model checkpoints (not included)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/Areeba-433/retinal-disease-detection.git
cd retinal-disease-detection
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install timm opencv-python albumentations scikit-learn pandas numpy matplotlib seaborn gradio grad-cam tqdm
```

### 4. Download the dataset

- Go to [ODIR-5K on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- Download and extract into `data/odir/`
- Final structure should be:

```
data/
└── odir/
    ├── ODIR-5K/
    │   └── ODIR-5K/         ← fundus images here
    └── full_df.csv
```

---

## 🚀 Training

```bash
cd src
python train.py
```

Expected output:

```
Using device: cpu
Total samples: 6392
Train: 5433 | Val: 959

Epoch 1/10 | Train Loss: 0.3227 | Val Loss: 0.2921 | Mean AUC: 0.7964
  Normal        : AUC = 0.7398
  Diabetes      : AUC = 0.7231
  Glaucoma      : AUC = 0.8544
  Cataract      : AUC = 0.8975
  Myopia        : AUC = 0.9610
  ✓ Saved best model (AUC=0.7964)

...

Epoch 10/10 | Train Loss: 0.1526 | Val Loss: 0.3002 | Mean AUC: 0.8447
Training complete! Best Mean AUC: 0.8452
```

---

## 🌐 Web Demo

```bash
cd src
python app.py
```

Opens at `http://localhost:7860` with a **public shareable link** valid for 72 hours.  
Upload any retinal fundus image and get instant predictions for all 8 diseases.

---

## 🔍 GradCAM Explainability

```bash
cd src
python evaluate.py "path/to/retinal/image.jpg"
```

Generates heatmaps overlaid on the original image showing which retinal regions the model focused on for each disease prediction. Saved to `outputs/gradcam_result.png`.

---

## 📈 Generate Charts

```bash
cd src
python -c "
from utils import plot_auc_bars
aucs = [0.7787, 0.7992, 0.8760, 0.9222, 0.8739, 0.8223, 0.9826, 0.7026]
plot_auc_bars(aucs)
"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch 2.12 |
| Model Backbone | EfficientNet-B0 via `timm` |
| Image Preprocessing | OpenCV (CLAHE) |
| Augmentation | Albumentations |
| Evaluation | scikit-learn (AUC-ROC) |
| Visualization | Matplotlib · Seaborn |
| Web Demo | Gradio |
| Explainability | GradCAM (`grad-cam`) |

---

## 📦 Dataset

[ODIR-5K — Ocular Disease Intelligent Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

- **6,392** fundus retinal images
- Left and right eye per patient
- **8 disease categories** with multi-label annotations
- Labels: Normal · Diabetes · Glaucoma · Cataract · AMD · Hypertension · Myopia · Other

---

## 🔮 Next Steps

- [ ] GradCAM heatmap visualization polish
- [ ] Temperature scaling for calibrated confidence scores  
- [ ] Upgrade to EfficientNet-B4 for higher AUC (~0.90+)
- [ ] Docker containerization for deployment
- [ ] APTOS 2019 dataset integration for DR severity grading (0–4)
- [ ] Federated learning across simulated hospital nodes

---

## 👩‍💻 Author

**Areeba Mumtaz Minhas**  
4th Year Software Engineering · PUCIT, University of the Punjab · Lahore, Pakistan

- 🔗 [LinkedIn](www.linkedin.com/in/areeba-mumtaz-minhas-2779a8301)
- 🐙 [GitHub](https://github.com/Areeba-433)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
