import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from ultralytics import YOLO
import os


yolo_model = YOLO("model/best.pt")


NAMES = ['Benign', 'Early', 'Pre-B', 'Pro-B', 
         'Glioma', 'Meningioma', 'No Tumor', 'Pituitary']



def detect_objects(image_path):
    """
    Runs YOLOv8 detection and returns the results object.
    """
    results = yolo_model(image_path)[0]
    return results



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])



def classify_roi(roi_crop, full_image, model, device):
    """
    Takes a cropped ROI and the full image, processes them,
    and predicts cancer name and type using the Combined Model.
    """
    try:
        
        roi_tensor = transform(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        
        full_tensor = transform(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        
        with torch.no_grad():
            if hasattr(model, "forward") and "tuple" in str(model.forward.__annotations__):
                outputs = model((roi_tensor, full_tensor))
            else:
                try:
                    outputs = model(roi_tensor, full_tensor)
                except Exception:
                    outputs = model(roi_tensor)  # fallback for simple models

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        
        cancer_name = NAMES[pred_idx] if pred_idx < len(NAMES) else "Unknown"

       
        if cancer_name in ['Benign', 'Early', 'Pre-B', 'Pro-B']:
            cancer_type = "Leukemia"
        elif cancer_name in ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']:
            cancer_type = "Brain Tumor"
        else:
            cancer_type = "Unknown"

        return cancer_name, cancer_type

    except Exception as e:
        print("Classification error:", e)
        return "Error", "Error"



if __name__ == "__main__":
    # quick standalone test
    img_path = "test_image.jpg"
    if os.path.exists(img_path):
        print("Running YOLO + Combined Model test...")
        results = detect_objects(img_path)
        print("Detected classes:", [NAMES[int(b.cls[0])] for b in results.boxes])
