import gradio as gr
import torch
import cv2
import numpy as np
import os
from model import ODIRModel
from dataset import get_transforms, apply_clahe

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

DEVICE = "cpu"
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract',
                 'AMD', 'Hypertension', 'Myopia', 'Other']

# Load trained model
model = ODIRModel(num_classes=8, pretrained=False)
model.load_state_dict(torch.load(
    "../outputs/best_model.pth", map_location=DEVICE))
model.eval()
transform = get_transforms("val")


def predict(image):
    if image is None:
        return {}
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = apply_clahe(img)
    tensor = transform(image=img)["image"].unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).numpy()[0]
    return {name: float(prob) for name, prob in zip(DISEASE_NAMES, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retinal Fundus Image"),
    outputs=gr.Label(num_top_classes=8, label="Disease Probabilities"),
    title="🔬 Retinal Disease Detector",
    description="Upload a fundus retinal image to detect 8 eye diseases using EfficientNet-B0 trained on ODIR-5K. Built by Areeba Mumtaz Minhas.",
    examples=[],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True)