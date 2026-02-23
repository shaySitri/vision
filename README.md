# Applied Computer Vision: Video Tracking & Facial Attribute Classification

[![Jupyter Notebook](https://img.shields.io/badge/Notebook-LFW_Siamese.ipynb-orange?logo=jupyter)]([https://github.com/shaySitri/LFW-Siamese/blob/main/LFW_Siamese.ipynb](https://colab.research.google.com/drive/1xlWDNxGmS6DW65X3ZL-YowezcdISLX5b?usp=sharing))

This project demonstrates practical computer vision pipelines implemented using modern deep learning frameworks.

The notebook includes:

* 🎥 Video Object Detection & Tracking
* 👤 Face Recognition in Video
* 🧔 Facial Attribute Classification (Beard / Earrings)
* 🔁 Transfer Learning with ResNet50
* 📊 Model Evaluation with Confusion Matrices & Classification Reports

---

## 🔍 Project Overview

This repository contains a Jupyter notebook implementing multiple computer vision tasks:

### 1️⃣ Video Object Detection & Tracking

* RetinaNet model (via ImageAI)
* Frame-by-frame detection
* Bounding box visualization
* Processed video generation with original audio reattached

**Technologies:**

* ImageAI
* OpenCV
* MoviePy

---

### 2️⃣ Face Recognition & Tracking

* Facial encoding using `face_recognition`
* Identity matching across frames
* Real-time bounding box labeling

**Technologies:**

* face_recognition
* OpenCV

---

### 3️⃣ Facial Attribute Classification

Binary classification tasks:

* Beard detection
* Earrings detection

Two approaches were implemented:

#### 🅰 Feature Extraction + CatBoost

* ResNet50 as pretrained feature extractor
* Image embeddings fed into CatBoostClassifier
* Evaluation using confusion matrix and classification report

#### 🅱 Transfer Learning (FastAI)

* `vision_learner(resnet50)`
* Fine-tuning pretrained CNN
* Performance evaluation and visualization

---

## 🎥 Results Preview

### Object Detection Example

![Detection Preview](assets/detection_result.png)

---

### Facial Attribute Classification

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🚀 Run the Full Notebook

The complete pipeline (including video processing) can be executed in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](PASTE_YOUR_COLAB_LINK_HERE)

---

## 🧠 Technical Skills Demonstrated

* Transfer Learning
* CNN Feature Extraction
* Video Frame Processing
* Face Recognition Pipelines
* Model Evaluation & Metrics
* Multi-framework CV workflow (ImageAI + PyTorch + FastAI + CatBoost)

---

## 📦 Requirements

```bash
pip install imageai face_recognition catboost fastai torch torchvision moviepy opencv-python scikit-learn
```

---

## 📌 Notes

* Video files are not included due to size constraints.
* Dataset files are not included (Kaggle required).
* The project is structured as a demonstration notebook.

---

# 🧠 עכשיו שאלה אסטרטגית חשובה

את רוצה:

* שזה יהיה ריפו עצמאי?
  או
* להכניס תחת repository של "Computer Vision Coursework"?

לפורטפוליו ML — אני ממליץ ריפו עצמאי.

---

אם תרצי, אני יכול גם:

* לנסח bullet חזק לקו"ח מתוך הפרויקט הזה
* להחליט אם זה נכנס לקו"ח או רק לגיטהאב
* לשדרג את ה-README לרמה אפילו יותר מרשימה

מה המטרה שלך כרגע — חיזוק ל-ML roles או בעיקר לסגור קצוות בגיטהאב?
