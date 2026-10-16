# 🌤️ Weather Image Classification

This repository contains a Machine Learning / Deep Learning project aimed at classifying images based on weather conditions. The implementation was done in a **Jupyter Notebook** using the **TensorFlow/Keras** library.

In this project, a custom Convolutional Neural Network (Baseline Custom CNN) is compared with a pre-trained model (MobileNetV2) utilizing Transfer Learning.

## 📊 Dataset

The data is sourced from the [Weather Dataset on Kaggle](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) (6,862 images).
The images are categorized into the following **11 classes**:
`dew`, `fogsmog`, `frost`, `glaze`, `hail`, `lightning`, `rain`, `rainbow`, `rime`, `sandstorm`, `snow`.

### Preprocessing

* **Image Resolution**: Resized to 128x128 pixels.
* **Normalization**: Pixel values scaled to the [0, 1] range (divided by 255.0).
* **Batch Size**: 32
* **Data Split**:
  * 70% Training
  * 10% Testing
  * 20% Validation

## 🧠 Model Architectures

### 1. Baseline CNN (Custom Model)

A simple, custom Convolutional Neural Network used as a baseline.

* **Structure**: 3 Convolutional Layers, 3 MaxPooling Layers.
* **Regularization & Overfitting Prevention**: Use of Dropout (0.25 and 0.5) and L2 Regularization.
* **Classifier**: GlobalAveragePooling2D and Dense layers with a final Softmax activation function (11 classes).
* **Callbacks**: Early Stopping applied (patience=10) with `restore_best_weights=True`.

### 2. MobileNetV2 (Transfer Learning)

Utilizing the **MobileNetV2** architecture (pre-trained on ImageNet) for feature extraction.

* The base layers of the model were frozen (`trainable = False`).
* A small custom classifier was added on top (GlobalAveragePooling2D -> Dense(128, ReLU) -> Dense(11, Softmax)).

## 📈 Results & Performance

* **Baseline Custom CNN**: Achieved an accuracy of approximately **~78%-83%** on the Test Dataset.
* **MobileNetV2**: Achieved an impressive accuracy of **~92.5%** in just 10 training epochs.

**Conclusions:**
As expected, the pre-trained model (MobileNetV2) significantly outperforms the custom one, thanks to the robust features it learned from ImageNet. The Baseline CNN is a good starting point, but it requires additional layers, hyperparameter tuning, and possibly Data Augmentation to further improve its performance.

## 🛠️ Installation and Requirements

To run this project locally, you will need Python 3.x and the following libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn seaborn pillow kagglehub
```

## 🚀 Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/Agony-sol/image-classification-cnn.git
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "CNN3_Fin.ipynb"
   ```
3. Run the cells. The dataset will be downloaded automatically via `kagglehub` (or you can download it manually if there are compatibility issues, as mentioned in the notebook).
