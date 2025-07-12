# Image-of-Cats-and-Dogs-Using-CNN


```markdown
# 🐶🐱 Cats vs Dogs Image Classification

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as either cats or dogs. The model is trained on the popular **Dogs vs Cats** dataset from Kaggle and demonstrates end-to-end implementation from data loading to prediction and visualization.

## 📂 Dataset

Downloaded from Kaggle via OpenDatasets:
- [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Directory structure:
  ```
  /content/dogs-vs-cats/
  ├── train/
  │   ├── cats/
  │   └── dogs/
  └── test/
  ```

## 🧠 Model Architecture

- Three convolutional layers with increasing filters (32 → 64 → 128)
- MaxPooling after each Conv layer
- Fully connected Dense layers with Dropout for regularization
- Sigmoid output layer for binary classification

## ⚙️ Training

- Input images resized to `(256, 256)`
- Normalization applied
- Model compiled with:
  - Optimizer: `Adam`
  - Loss: `Binary Crossentropy`
  - Metric: `Accuracy`
- Trained for 3 epochs using TensorFlow `image_dataset_from_directory`

## 📈 Evaluation

Accuracy and Loss plots generated for training and validation sets. Example prediction is shown on a sample image with the predicted class.

## 🔮 Prediction

Load an image from the dataset or external source, preprocess it, and run model inference using:
```python
model.predict(...)
```

## 💾 Save & Deployment

- Trained model saved to disk as `catvsdog.h5`
- Ready for deployment via Streamlit or Flask

## 🚀 Next Steps
- Streamlit integration for web-based UI
- Data augmentation for better generalization
- Grad-CAM for visual explanation of predictions
## 📚 Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy, Pandas, Matplotlib
- OpenDatasets
Install dependencies:
```bash
pip install tensorflow opendatasets numpy pandas matplotli
Created by [Vinamre] 👨‍💻  
A project that blends precision with creativity.
Want to include Streamlit instructions or link it to a GitHub repo next?
