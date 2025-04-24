# Finger Vein Identification System using Fast DenseNet

This project is a deep learning-based biometric identification system that uses finger vein images for recognizing individuals. It features a GUI application for dataset exploration, image preprocessing, model training, threshold tuning, and real-time identification.

##  Features

- **GUI Application** built with Tkinter for ease of use.
- **Dataset Exploration** with visualization of image distributions.
- **Image Preprocessing** pipeline using contrast enhancement and CLAHE.
- **Fast DenseNet Model** tailored for efficient training and high accuracy.
- **Two-stage Training**: Initial training and fine-tuning phases.
- **Confidence Thresholding** to reject low-confidence predictions as "Unknown".
- **Visualization Tools**: Training history, prediction confidence, confusion matrix, etc.
- **Model Saving and Loading** support for persistence and reuse.

##  Technologies Used

- **Python** 3.11
- **TensorFlow / Keras** for model training
- **OpenCV** for image preprocessing
- **Matplotlib & Seaborn** for plotting
- **Tkinter** for GUI
- **NumPy** for data manipulation
- **Scikit-learn** for train-test splitting and evaluation



##  Dataset Requirements

- Structure: Each subfolder should follow the naming pattern `vein<personID>_<sessionID>` (e.g., `vein001_1`).
- Supported image formats: `.png`, `.jpg`, `.jpeg`
- Minimum of 2 classes (people) recommended.

##  How to Use

1. **Install Dependencies**:
   
   pip install tensorflow opencv-python matplotlib seaborn scikit-learn pillow
   

2. **Run the Application**:
   
   python modified-finger-vein-code_1.py
   

3. **Steps in the GUI**:
   - Upload a dataset using `Upload Dataset`.
   - Click `Preprocess Data` to convert raw images.
   - Adjust the `Confidence Threshold` slider if needed.
   - Click `Train Model` to start training.
   - After training, test images via `Identify Image`.
   - Use `Update Threshold` to reanalyze the model performance.
   - Load a previously trained model using `Load Model`.

## Model Architecture

- **Base Model**: DenseNet121 (ImageNet weights, last ~30 layers trainable)
- **Custom Top Layers**:
  - GlobalAveragePooling
  - Dense(256) -> Dropout(0.3)
  - Dense(128) -> Dropout(0.2)
  - Output: Softmax for classification

##  Confidence Threshold

- Allows marking low-confidence predictions as "Unknown".
- Default: `0.5` (can be adjusted via GUI)

##  Outputs

- Training accuracy/loss curves
- Confidence histograms
- Confusion matrix for confident predictions
- Logs of operations in the GUI

