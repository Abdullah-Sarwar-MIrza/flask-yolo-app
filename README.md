# 🧠 Intelligent Cancer Diagnosis System (FYP / MS Project)

## 📌 Overview

This project is an end-to-end deep learning–based cancer diagnosis system focused primarily on **leukemia detection**, with extensibility to **brain tumor analysis**. It integrates **YOLOv8** for automatic Region of Interest (ROI) detection and multiple **CNN-based classifiers** to classify detected regions. The system is deployed as a **Flask web application** for real-time medical image analysis.

This project was developed as a **Final Year Project (FYP) / MS Thesis**, aiming to bridge the gap between research-level deep learning models and practical, deployable medical AI systems.

---

## 🎯 Key Objectives

* Automatically detect abnormal regions in medical images
* Classify detected regions using deep learning models
* Compare performance of multiple CNN architectures
* Provide a real-time, user-friendly web interface
* Build a scalable foundation for future explainability and reporting features

---

## 🏗️ System Architecture

1. **Input Medical Image** (Blood smear / Brain MRI)
2. **YOLOv8 Detector**

   * Detects cancer-related regions
   * Crops ROIs for classification
3. **Classifier Models**

   * CNN (Baseline)
   * VGG16
   * ResNet18
   * ResNet50
   * EfficientNet
   * Hybrid / Combined Feature Model
4. **Flask Web Application**

   * Image upload
   * Model selection
   * Real-time prediction output

---

## 🧪 Models & Technologies Used

### 🔍 Detection

* **YOLOv8** (Ultralytics)

### 🧠 Classification

* Custom CNN
* VGG16
* ResNet18
* ResNet50
* EfficientNet
* Combined Feature Model (ROI + Full Image)

### 🌐 Backend & Deployment

* Flask (Python)
* PyTorch
* OpenCV
* NumPy

---

## 📂 Project Structure (Simplified)

```
├── app.py                  # Flask application
├── yolo.py                 # YOLOv8 detection logic
├── models/                 # Trained classifier models
├── utils.py                # Helper functions
├── templates/              # HTML files
├── static/                 # CSS / JS / images
├── datasets/               # YOLO & classification datasets
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Models

* Place YOLOv8 weights inside the appropriate folder
* Place trained classifier `.pth` files in the `models/` directory

### 4️⃣ Run the Flask App

```bash
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📊 Supported Modalities

* ✅ Leukemia (Blood Smear Images)
* ✅ Brain Tumor (MRI Images)

The system can be extended to additional cancer types with minimal changes.

---

## 📚 Important Resources & References

* YOLOv8 (Ultralytics): [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* PyTorch: [https://pytorch.org/](https://pytorch.org/)
* Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
* ResNet Paper: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
* EfficientNet Paper: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
* VGG Paper: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

---

## ⚠️ Disclaimer

This project is intended for **academic and research purposes only**. It is **not a certified medical diagnostic tool** and should not be used for clinical decision-making.

---

## 👨‍🎓 Author

**Abdullah Sarwar Mirza**
Final Year / MS Research Project

---

## ⭐ Acknowledgements

* Ultralytics YOLOv8 Team
* Open-source deep learning community
* Kaggle & open medical datasets

---

If you find this project useful, feel free to ⭐ the repository!
