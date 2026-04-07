import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Configuration
DEVICE = torch.device("cpu")
IMG_SIZE = 224

# Transformation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Initialize FastAPI
app = FastAPI(title="Brain Tumor Detection API")
templates = Jinja2Templates(directory="templates")

# Globals for Grad-CAM
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Load Models
print("Loading Stage 1 Model...")
stage1 = models.resnet18(pretrained=False)
stage1.fc = nn.Linear(stage1.fc.in_features, 1)
try:
    stage1.load_state_dict(torch.load("stage1_tumor_detector.pth", map_location=DEVICE))
except Exception as e:
    print(f"Failed to load stage1: {e}")
stage1.to(DEVICE)
stage1.eval()

# Register Hooks on Stage 1
stage1.layer4.register_forward_hook(forward_hook)
stage1.layer4.register_backward_hook(backward_hook)

print("Loading Stage 2 Model...")
stage2_classes = ['glioma', 'meningioma', 'pituitary']
stage2 = models.resnet50(pretrained=False)
stage2.fc = nn.Linear(stage2.fc.in_features, len(stage2_classes))
try:
    stage2.load_state_dict(torch.load("stage2_tumor_classifier.pth", map_location=DEVICE))
except Exception as e:
    print(f"Failed to load stage2: {e}")
stage2.to(DEVICE)
stage2.eval()


def generate_grad_cam(model, output_tensor, original_img_pil):
    global activations, gradients
    model.zero_grad()
    output_tensor.backward(retain_graph=True)
    
    if len(gradients) == 0 or len(activations) == 0:
        return None

    grads = gradients[-1]            
    acts = activations[-1]
    
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()
    cam = F.relu(cam)
    
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam / cam.max()
    
    img_arr = np.array(original_img_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_arr, 0.6, heatmap, 0.4, 0)
    
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if is_success:
        return base64.b64encode(buffer).decode("utf-8")
    return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    global activations, gradients
    activations.clear()
    gradients.clear()

    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    # 1. Stage 1: Tumor Detection
    stage1_output = stage1(img_tensor)
    prob_tumor = torch.sigmoid(stage1_output).item()
    
    if prob_tumor < 0.5:
        return {
            "has_tumor": False,
            "confidence": round((1 - prob_tumor) * 100, 2),
            "message": "No tumor detected.",
            "type": None,
            "overlay_base64": None
        }

    # Generate Grad-CAM for Stage 1
    gradcam_b64 = generate_grad_cam(stage1, stage1_output, img_pil)

    # 2. Stage 2: Tumor Classification
    with torch.no_grad():
        stage2_output = stage2(img_tensor)
        probs = torch.softmax(stage2_output, dim=1)
        conf_class, pred_class = torch.max(probs, 1)
        tumor_type = stage2_classes[pred_class.item()]

    return {
        "has_tumor": True,
        "confidence": round(prob_tumor * 100, 2),
        "message": "Tumor detected.",
        "type": tumor_type.capitalize(),
        "type_confidence": round(conf_class.item() * 100, 2),
        "overlay_base64": gradcam_b64
    }
