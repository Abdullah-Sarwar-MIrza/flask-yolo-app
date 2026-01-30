from flask import Flask, render_template, request, redirect
import os
import cv2
from ultralytics import YOLO
from db_config import create_connection, insert_detection


# import classifier helper
from utils import classify_roi  # function added in utils.py

import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# YOLO model (unchanged)
model = YOLO('model/best.pt')


# Load combined classifier model (do not replace YOLO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model = None
combined_model_path = "model/best_combined_model.pth"
if os.path.exists(combined_model_path):
    try:
        combined_model = torch.load(combined_model_path, map_location=device)
        # if torch.load returned a state-dict instead of a model object we cannot use it directly.
        # We keep it tolerant: if it's not callable later, classify_roi will handle it.
        if hasattr(combined_model, "eval"):
            combined_model.eval()
        print("Combined model loaded from", combined_model_path)
    except Exception as e:
        print("Warning: Failed to load combined model (torch.load):", e)
        combined_model = None
else:
    print("Combined model not found at", combined_model_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # Run YOLO detection (unchanged)
    results = model(upload_path)[0]
    detections = []

    img = cv2.imread(upload_path)
    h_img, w_img = img.shape[:2]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

        # clip bbox to image size
        xmin = max(0, min(xmin, w_img - 1))
        xmax = max(0, min(xmax, w_img - 1))
        ymin = max(0, min(ymin, h_img - 1))
        ymax = max(0, min(ymax, h_img - 1))

        # draw box & label 
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Insert detection into DB 
        try:
            conn = create_connection()
            insert_detection(conn, filename, label, conf, xmin, ymin, xmax, ymax)
            conn.close()
        except Exception as e:
            print("DB insert warning:", e)

        
        roi_crop = None
        cancer_name, cancer_type = "Unknown", "Unknown"
        try:
            # If bbox is invalid (zero area), skip classification
            if xmax > xmin and ymax > ymin:
                roi_crop = img[ymin:ymax, xmin:xmax].copy()
                if combined_model is not None:
                    try:
                        cancer_name, cancer_type = classify_roi(roi_crop, img, combined_model, device)
                    except Exception as e:
                        print("Classification warning:", e)
                        cancer_name, cancer_type = "Unknown", "Unknown"
                else:
                    # Model not loaded
                    cancer_name, cancer_type = "NotAvailable", "NotAvailable"
            else:
                cancer_name, cancer_type = "InvalidBox", "InvalidBox"
        except Exception as e:
            print("ROI crop warning:", e)
            cancer_name, cancer_type = "Unknown", "Unknown"
       

        detections.append({
            'label': label,
            'confidence': round(conf, 2),
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            
            'cancer_name': cancer_name,
            'cancer_type': cancer_type
            
        })

   
    result_filename = 'result_' + filename
    result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    cv2.imwrite(result_path, img)

    return render_template('index.html', result=result_filename, original=filename, detections=detections)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
