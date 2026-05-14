# Brain Tumor Classification using Deep Learning

This project builds and compares multiple deep learning models to classify brain MRI images into four categories:

Glioma
Meningioma
Pituitary tumor
No tumor

We use both a custom CNN and transfer learning models (VGG16 and ResNet50) to evaluate performance.

## Dataset

The dataset is organized into training and testing folders:

Brain_Tumor_Dataset/
 ├── Training/
 └── Testing/

Each folder contains four classes:

glioma
meningioma
pituitary
notumor

## Total samples:

Training: 4571 images
Validation: 1141 images
Testing: 1311 images

## Preprocessing & Augmentation

Images are:

Resized to 128 × 128
Normalized (rescaled to [0,1])
Augmented using:
Rotation
Zoom
Horizontal flip
Width/height shift
## Models Used
1. Custom CNN (Baseline)

A deep convolutional neural network with:

4 Conv layers (32 → 256 filters)
Batch Normalization
MaxPooling
Dropout regularization
Dense classifier (Softmax)
2. Transfer Learning - VGG16
Pretrained on ImageNet
Frozen convolutional base
Custom dense head (256 neurons + dropout)
3. Transfer Learning - ResNet50
Pretrained on ImageNet
Global Average Pooling
Dense + Dropout classifier
## Results
Model	Test Accuracy
CNN Baseline	~79.8%
VGG16	~81.9%
ResNet50	~51.6%

✔ Best performing model: VGG16

## Observations
VGG16 performed best in terms of accuracy and stability.
CNN baseline achieved strong performance without pretraining.
ResNet50 underperformed likely due to:
Limited fine-tuning
Small image resolution (128×128)
## Evaluation Metrics

Each model is evaluated using:

Accuracy
Precision / Recall / F1-score
Confusion Matrix
ROC Curves (per class)
## Training Details
Optimizer: Adam (lr = 1e-4)
Loss function: Categorical Crossentropy
Batch size: 32
Epochs: 15–20
Callbacks:
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint
## Tech Stack
Python 
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn
OpenCV
Google Colab
## How to Run
Clone the repository:
git clone https://github.com/your-username/brain-tumor-classification.git
Install dependencies:
pip install -r requirements.txt
Run the notebook:
jupyter notebook brain_tumor_classification.ipynb

## Sample Output
Training accuracy/loss curves
Confusion matrices
ROC curves per class
Sample image predictions
## Future Improvements
Fine-tune ResNet50 instead of freezing all layers
Try EfficientNet / DenseNet
Increase image resolution (224×224)
Add Grad-CAM explainability for medical interpretation
Deploy as a web app (Streamlit / Flask)
