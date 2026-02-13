# 🔥 Forest Fire AI Classifier

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey.svg)](https://flask.palletsprojects.com/)

An end-to-end Deep Learning solution using **Bidirectional LSTMs** to classify wildfire risks based on the UCI Forest Fires meteorological dataset.

## ✨ Key Features
- **🧠 Advanced RNN Architecture**: Utilizes Bidirectional LSTMs with Batch Normalization for robust feature extraction from tabular data.
- **🌐 Interactive Dashboard**: A premium, responsive Web UI built with Flask and Glassmorphism design aesthetics.
- **🤗 Hugging Face Powered**: native Gradio-based interface included for seamless deployment to Hugging Face Spaces.
- **⌨️ Smart CLI**: Interactive terminal prediction script that allows for real-time risk analysis.
- **📊 Performance Ready**: Includes preprocessing pipelines, training curves, and automated model evaluation.

## 📂 Project Architecture
```text
C:\RNN2
├── app.py              # Flask Web Application Backend
├── train.py            # Optimized Model Training Pipeline
├── predict.py          # Interactive CLI Prediction Tool
├── preprocess.py       # Data Cleaning & Feature Engineering
├── data/               # Scaling & Feature Metadata
├── models/             # Exported RNN Models (.keras)
├── static/             # Web Assets (CSS/JS)
├── templates/          # HTML Interface
└── results/            # Performance Visualizations
```

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/snehadm-25/Forest-Fire-RNN.git
cd Forest-Fire-RNN
pip install -r requirements.txt
```

### 2. Prepare & Train
Generate preprocessed data and train the Bidirectional LSTM model:
```bash
python preprocess.py
python train.py
```

### 3. Launch the Web App
Start the diagnostic server:
```bash
python app.py
```
Visit **`http://127.0.0.1:5001`** to use the interactive dashboard.

### 4. Terminal Inference
Run the interactive prediction tool:
```bash
python predict.py
```

## 🛠️ Tech Stack
- **Core**: Python 3.11, TensorFlow, Keras
- **Data**: Pandas, NumPy, Scikit-Learn
- **Web**: Flask, Vanilla JS, Premium CSS
- **Visualization**: Matplotlib, Seaborn

---
*Created as part of an end-to-end RNN implementation project.*