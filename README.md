# Facial Emotion Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify facial expressions into one of seven emotion categories using the FER2013 dataset. It is developed as part of the course **DS677 - Deep Learning** at NJIT (Spring 2025).

---

## 🚀 Project Overview

Facial emotion recognition plays a vital role in human-computer interaction and mental health monitoring. This project:
- Uses CNNs to extract features from 48x48 grayscale facial images.
- Classifies images into 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- Demonstrates real-time prediction with an inference demo.

---

## 📂 Directory Structure

```
Facial-Emotion-Recognition/
├── data/
│   └── archive/
│       ├── train/
│       └── test/
├── src/
│   ├── models/
│   ├── data/
│   ├── utils/
│   └── train.py
├── demo/
│   └── predict.py
├── outputs/
├── notebooks/
│   └── eda.ipynb
├── report/
│   ├── final_report.pdf
│   └── presentation_slides.pptx
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

- **Name:** FER2013 (Facial Expression Recognition 2013)
- **Source:** [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🧠 Model Architecture

- Input: 48×48 grayscale image
- Conv2D → ReLU → MaxPooling → Dropout
- 2 Convolutional blocks followed by:
- Fully Connected Layer → Dropout → Output Layer (7 classes)

---

## 🛠️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python3 src/train.py
```

### 4. Run a demo prediction
```bash
python3 demo/predict.py
```

---

## 📈 Results

- **Training Accuracy:** ~96.67%


---

## 💡 Applications

- Emotion-aware chatbots
- Online mental health platforms
- Smart classrooms
- Driver fatigue detection

---

## 👥 Team Members

- **Ganesh** – Model architecture and training
- **Vijaya** – Data preprocessing and evaluation
- **Jyotsna** – Documentation and GitHub deployment



---

## 📜 License

MIT License – feel free to reuse with attribution.