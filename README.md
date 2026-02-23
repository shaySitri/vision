# Applied Computer Vision: Video Tracking & Facial Attribute Classification

[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Applied_Computer_Vision.ipynb-orange?logo=jupyter)]([https://github.com/shaySitri/LFW-Siamese/blob/main/LFW_Siamese.ipynb](https://colab.research.google.com/drive/1xlWDNxGmS6DW65X3ZL-YowezcdISLX5b?usp=sharing))

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

**🎥 Results Preview**

![Detection Preview](res_prev/object_detection1.jpg)

![Detection Preview](res_prev/object_detection2.jpg)
---

### 2️⃣ Face Recognition & Tracking

* Facial encoding using `face_recognition`
* Identity matching across frames
* Real-time bounding box labeling

**Technologies:**

* face_recognition
* OpenCV

**🎥 Results Preview**
![Detection Preview](res_prev/face_detection1.jpg)

Original Video **[here](https://www.youtube.com/watch?v=W5PRZuaQ3VM)**

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
