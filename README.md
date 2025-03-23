# FalseLight Vision 20M 🕵️‍♂️🧠  
*A Transfer Learning-Based Deepfake Detection Model Using Xception*

## 📌 Overview

**FalseLight Vision 20M** is a deep learning model developed for the detection of AI-generated facial forgeries (deepfakes). It leverages **transfer learning** using the **Xception CNN architecture** and is trained on a diverse dataset of real and manipulated images. Designed for high accuracy and real-time usability, FalseLight integrates into an interactive web application that can classify images and videos with visual feedback.

---

## Download .h5 Model
Download from [HuggingFace](https://huggingface.co/nezuno177/FalseLight-Vision-20M/blob/main/FalseLight%20Vision%2020M.h5)

---

## 🧠 Model Architecture

- **Base Architecture**: Xception (pre-trained on ImageNet)
- **Modified Output Layers**:
  - Global Average Pooling
  - Dropout (rate = 0.5)
  - Dense (1 unit, sigmoid activation)
- **Training Strategy**:
  - Phase 1: Freeze base, train top layers
  - Phase 2: Unfreeze upper base layers, fine-tune entire model

**Total Parameters**: ~20.86M  
**Model Size**: ~146 MB

---

## 🧪 Dataset

- **Source**: [DeepfakeBench](https://github.com/DeepfakeBench)
- **Total Images**: 150,000 
  - 75,000 real  
  - 75,000 fake  
- **Preprocessing**:
  - Face detection
  - Resize to 224x224
  - Xception-style normalization
- **Data Augmentation**:
  - Horizontal flip
  - Random rotation
  - Zoom & shift
  - Contrast jitter

---

## 📊 Performance

| Metric           | Value     |
|------------------|-----------|
| **Accuracy**     | 0.97     |
| **Precision**    | 0.96     |
| **Recall**       | 0.99     |
| **F1-Score**     | 0.97     |
| **AUC**          | 0.94     |
| **Inference Time (GPU)** | ~0.06s/image |

Tested on a held-out test set (62,000 images, 50/50 real/fake).

---

## 🧪 Benchmark Comparison

| Model            | Accuracy | Parameters |
|------------------|----------|------------|
| **FalseLight (Xception)** | **97%**   | 20.8M      |
| EfficientNet-B4  | 84%      | 19M        |
| ResNet-101       | 82%      | 44.5M      |
| DenseNet-121     | 83%      | 7M         |
| MobileNetV2      | 78%      | 3.4M       |

---

## 🚀 Deployment: Real-Time Web App

Built using **FastAPI + HTML/JS** frontend.

### 🌐 Features:
- Upload **images** or **videos**
- Detect multiple faces per frame
- Real-time prediction with bounding boxes
- Visual output includes:
  - Face bounding boxes
  - "Fake: X%" probability labels
- Streaming mode for live video analysis

### 🛠 Tech Stack:
- TensorFlow / Keras
- FastAPI (Python backend)
- face_recognition (HOG-based face detector)
- OpenCV for video processing
- HTML / JavaScript frontend

---


## 📚 References

- Yan, Z. et al. (2024). *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*. NeurIPS.
- Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions*. CVPR.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Rössler, A. et al. (2019). *FaceForensics++*. ICCV.

---

## 🧑‍💻 Author

**Zhou Hui**  
Final Year Student – CEG3001 Capstone Project  
Singapore Institute of Technology  
Supervised by: *Ms Ng Pai Chet*
With help from: *Gabriel Lee Jun Rong* & *Elmo Cham Rui An*

---

## 📬 Citation

```
@misc{falselight2025,
  title = {FalseLight Vision 20M: A Transfer Learning-Based Approach for Deepfake Detection Using Xception},
  author = {Zhou Hui, Gabriel Lee Jun Rong, Elmo Cham Rui An},
  year = {2025},
  institution = {Singapore Institute of Technology}
}
```

---

## 🤝 Contributions & License

Open to collaboration. Pull requests and suggestions are welcome!
