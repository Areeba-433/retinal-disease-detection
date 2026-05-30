# 🔬 Retinal Disease Detection using Deep Learning

A multi-label eye disease classifier that detects **8 diseases simultaneously** from fundus retinal images using EfficientNet-B0. Trained on the ODIR-5K dataset with a Mean AUC of **0.8452**.

Built by **Areeba Mumtaz Minhas** — 3rd year Software Engineering student at PUCIT.

---

## 🎯 Diseases Detected

| Code | Disease |
|------|---------|
| N | Normal |
| D | Diabetic Retinopathy |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration (AMD) |
| H | Hypertension |
| M | Myopia |
| O | Other |

---

## 📊 Results (10 Epochs, CPU Training)

| Disease | AUC |
|---------|-----|
| Myopia | **0.9826** |
| Cataract | **0.9222** |
| Glaucoma | **0.8760** |
| AMD | **0.8739** |
| Hypertension | **0.8223** |
| Diabetes | **0.7992** |
| Normal | **0.7787** |
| Other | **0.7026** |
| **Mean AUC** | **0.8452** |

> Trained entirely on CPU — no GPU required.

---

## 🏗️ Architecture
Input (Fundus Image)
↓
CLAHE Contrast Enhancement
↓
Albumentations Augmentation (flip, rotate, color jitter)
↓
EfficientNet-B0 Backbone (ImageNet pretrained, fine-tuned)
↓
Multi-label Classification Head (8 outputs)
↓
BCE with Logits Loss + Per-disease AUC-ROC
**Key design decisions:**
- **EfficientNet-B0** — best accuracy/speed tradeoff for CPU training
- **CLAHE preprocessing** — enhances vessel and lesion visibility in retinal images
- **Multi-label BCE loss** — one image can have multiple diseases simultaneously
- **Per-disease AUC-ROC** — standard clinical evaluation metric

---

## 📁 Project Structure
retinal-disease-detection/
├── src/
│   ├── dataset.py       # Data loading + CLAHE preprocessing
│   ├── model.py         # EfficientNet backbone + classifier head
│   ├── train.py         # Training loop + AUC evaluation
│   ├── evaluate.py      # GradCAM heatmap generation
│   ├── app.py           # Gradio web demo
│   └── utils.py         # ROC curves, loss plots, charts
├── data/                # Dataset (not included, see Setup)
├── outputs/             # Saved model checkpoints (not included)
├── notebooks/
│   └── explore.ipynb    # Data exploration notebook
└── requirements.txt
---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/retinal-disease-detection.git
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

### 4. Download dataset
- Go to [ODIR-5K on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- Download and extract into `data/odir/`
- Structure should be:
data/
└── odir/
├── ODIR-5K/
│   └── ODIR-5K/        ← images here
└── full_df.csv
---

## 🚀 Training

```bash
cd src
python train.py
```

Expected output:
Using device: cpu
Total samples: 6392
Train: 5433 | Val: 959
Epoch 1/10 | Train Loss: 0.3227 | Val Loss: 0.2921 | Mean AUC: 0.7964
Myopia        : AUC = 0.9610
Cataract      : AUC = 0.8975
...
Training complete! Best Mean AUC: 0.8452
---

## 🌐 Run Web Demo

```bash
cd src
python app.py
```

Opens at `http://localhost:7860` with a public shareable link. Upload any fundus image and get instant predictions for all 8 diseases.

---

## 🔍 GradCAM Explainability

```bash
cd src
python evaluate.py "path/to/retinal/image.jpg"
```

Generates heatmaps showing which regions of the retina the model focused on for each disease prediction. Saved to `outputs/gradcam_result.png`.

---

## 📈 Generate Charts

```bash
cd src
python -c "
from utils import plot_auc_bars
import numpy as np
aucs = [0.7787, 0.7992, 0.8760, 0.9222, 0.8739, 0.8223, 0.9826, 0.7026]
plot_auc_bars(aucs)
"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch |
| Model Backbone | EfficientNet-B0 (timm) |
| Image Processing | OpenCV, Albumentations |
| Evaluation | scikit-learn (AUC-ROC) |
| Visualization | Matplotlib, Seaborn |
| Web Demo | Gradio |
| Explainability | GradCAM (grad-cam) |

---

## 🌍 Why This Matters

- **450 million+** people worldwide live with diabetes
- Diabetic retinopathy is the **leading cause of blindness** in working-age adults
- Most cases are **preventable with early detection**
- AI-assisted screening can help reach patients in under-resourced areas where ophthalmologists aren't available

---

## 🔮 Next Steps

- [ ] GradCAM heatmap visualization polish
- [ ] Temperature scaling for calibrated confidence scores
- [ ] Upgrade to EfficientNet-B4 for higher AUC
- [ ] Docker containerization for deployment
- [ ] APTOS 2019 dataset integration for DR grading

---

## 👩‍💻 Author

**Areeba Mumtaz Minhas**
3rd Year Software Engineering Student — PUCIT, University of the Punjab
- LinkedIn: [your linkedin]
- GitHub: [your github]

---

## 📄 Dataset

[ODIR-5K — Ocular Disease Intelligent Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- 6,392 fundus images
- Left and right eye per patient
- 8 disease categories, multi-label annotations
