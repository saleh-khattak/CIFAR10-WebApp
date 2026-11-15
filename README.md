# 🖼️ CIFAR-10 Image Classification Web App

<p align="center">
  <strong>Deep Learning | PyTorch | Streamlit</strong>
</p>

<p align="center">
  A CNN-based image classification application that predicts objects from uploaded images using the CIFAR-10 dataset.
</p>

<p align="center">
  ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
  ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
  ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit)
  ![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
</p>

---

## 📌 Overview

**CIFAR-10 Image Classification Web App** is a deep learning project developed using **PyTorch** and **Streamlit**.

It uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 different object categories.

Users can upload an image through an interactive web interface and receive the predicted class, confidence score, and top 3 predictions.

## ✨ Features

- 📤 Upload images in JPG, JPEG, and PNG formats.
- 🤖 Predict image classes using a trained CNN model.
- 📊 Display prediction confidence percentages.
- 🏆 Show the top 3 predicted classes.
- 🖼️ Display uploaded images in the web interface.
- 💾 Save uploaded images locally.
- ⚡ Support CPU and CUDA-enabled GPU inference.
- 🔄 Automatically resize and preprocess images before prediction.

## 🗂️ CIFAR-10 Dataset

The CIFAR-10 dataset contains 60,000 color images of size 32 × 32 pixels, divided into 10 different classes.

The model recognizes the following categories:

| No. | Class |
|:---:|---|
| 1 | ✈️ Airplane |
| 2 | 🚗 Automobile |
| 3 | 🐦 Bird |
| 4 | 🐱 Cat |
| 5 | 🦌 Deer |
| 6 | 🐕 Dog |
| 7 | 🐸 Frog |
| 8 | 🐎 Horse |
| 9 | 🚢 Ship |
| 10 | 🚚 Truck |

## 🧠 CNN Model Architecture

The application uses a Convolutional Neural Network implemented with PyTorch.

| Layer | Configuration |
|---|---|
| Input | 3 × 32 × 32 RGB image |
| Conv Layer 1 | 3 → 32 filters, 3 × 3 kernel |
| Activation | ReLU |
| Pooling | MaxPool 2 × 2 |
| Conv Layer 2 | 32 → 64 filters, 3 × 3 kernel |
| Activation | ReLU |
| Pooling | MaxPool 2 × 2 |
| Conv Layer 3 | 64 → 128 filters, 3 × 3 kernel |
| Activation | ReLU |
| Pooling | MaxPool 2 × 2 |
| Flatten | 128 × 4 × 4 = 2048 |
| Fully Connected 1 | 2048 → 512 |
| Activation | ReLU |
| Output Layer | 512 → 10 classes |

### Model Workflow

```text
       Input Image
            |
            ▼
     Image Preprocessing
            |
            ▼
      Convolution Layer 1
            |
            ▼
       ReLU + MaxPool
            |
            ▼
      Convolution Layer 2
            |
            ▼
       ReLU + MaxPool
            |
            ▼
      Convolution Layer 3
            |
            ▼
       ReLU + MaxPool
            |
            ▼
          Flatten
            |
            ▼
      Fully Connected
            |
            ▼
       Output (10 Classes)
            |
            ▼
      Softmax Probabilities
            |
            ▼
    Predicted Class & Confidence
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image transformations and utilities |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |

## 📁 Project Structure

```text
CIFAR10-WebApp/
│
├── app.py                 # Streamlit application
├── model.pth              # Trained CNN model weights
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignored files
│
├── uploaded_images/       # Saved uploaded images
│
└── venv/                  # Virtual environment (not uploaded)
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/saleh-khattak/CIFAR10-WebApp.git
```

### 2. Navigate to the Project Directory

```bash
cd CIFAR10-WebApp
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🚀 How to Use

1. Open the Streamlit application.
2. Upload an image in JPG, JPEG, or PNG format.
3. The application automatically preprocesses the image.
4. The trained CNN model analyzes the image.
5. View the predicted class and confidence percentage.
6. Explore the top 3 predictions.

## 📦 Model Weights

The application requires the trained model weights file:

```text
model.pth
```

Place `model.pth` in the root directory alongside `app.py`.

**Note:** The model architecture in the application must match the architecture used during training.

## 🔮 Future Improvements

- Improve classification performance through model optimization.
- Add model evaluation metrics and visualizations.
- Display confusion matrices and classification reports.
- Enhance the web interface.
- Deploy the application online.

## 👨‍💻 Author

**Muhammad Saleh**

BS Computer Science Student  
University of Engineering and Technology (UET), Peshawar

<p>
  <a href="https://github.com/saleh-khattak">GitHub</a> •
  <a href="https://www.linkedin.com/in/muhammad-saleh-a842b434a/">LinkedIn</a>
</p>

---

<p align="center">
  <strong>Built with ❤️ using Python, PyTorch, and Streamlit.</strong>
</p>
