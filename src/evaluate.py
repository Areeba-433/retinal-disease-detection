import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import ODIRModel
from dataset import get_transforms
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISEASE_NAMES = ['Normal','Diabetes','Glaucoma','Cataract',
                 'AMD','Hypertension','Myopia','Other']

def load_model(checkpoint_path):
    model = ODIRModel(num_classes=8, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def predict_and_explain(image_path, checkpoint_path="../outputs/best_model.pth"):
    model = load_model(checkpoint_path)

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms("val")
    tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    # Get predictions
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # GradCAM on last conv layer
    target_layer = model.backbone.conv_head
    cam = GradCAM(model=model, target_layers=[target_layer])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Disease Detection + GradCAM Heatmaps", fontsize=14)

    img_resized = cv2.resize(image, (224, 224)) / 255.0

    for i, (name, prob) in enumerate(zip(DISEASE_NAMES, probs)):
        ax = axes[i // 4][i % 4]
        grayscale_cam = cam(input_tensor=tensor,
                            targets=None)[0]
        overlay = show_cam_on_image(img_resized.astype(np.float32),
                                    grayscale_cam, use_rgb=True)
        ax.imshow(overlay)
        ax.set_title(f"{name}\n{prob:.1%}", fontsize=10,
                     color='red' if prob > 0.5 else 'green')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("../outputs/gradcam_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPredictions:")
    for name, prob in zip(DISEASE_NAMES, probs):
        flag = "⚠️ DETECTED" if prob > 0.5 else "✓ Clear"
        print(f"  {name:<14}: {prob:.1%}  {flag}")

if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "../data/odir/ODIR-5K/0_right.jpg"
    predict_and_explain(img)