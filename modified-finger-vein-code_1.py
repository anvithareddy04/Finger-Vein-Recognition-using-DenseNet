import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from PIL import Image, ImageTk

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Global variables for paths and image size
DATASET_PATH = ""
OUTPUT_PATH = os.path.join(os.getcwd(), "finger_vein_models")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'models'), exist_ok=True)
image_size = (128, 128)  # Balanced image size for speed and accuracy

# Configure TensorFlow for better performance
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    # If no GPU available, optimize for CPU
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

# ------------------ Backend Functions ------------------

def explore_dataset():
    """Load dataset folders, group by person, plot distribution, and return groups."""
    global DATASET_PATH
    person_folders = [f for f in os.listdir(DATASET_PATH)
                      if os.path.isdir(os.path.join(DATASET_PATH, f)) and f.startswith('vein')]
    person_groups = {}
    for folder in person_folders:
        match = re.match(r'vein(\d+)_\d+', folder)
        if match:
            person_id = match.group(1)
            if person_id not in person_groups:
                person_groups[person_id] = []
            person_groups[person_id].append(folder)
    person_image_counts = {}
    for person_id, folders in person_groups.items():
        total_images = 0
        for folder in folders:
            folder_path = os.path.join(DATASET_PATH, folder)
            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(img_files)
        person_image_counts[f"Person {person_id}"] = total_images

    # Save plot to file instead of displaying directly
    fig = plt.figure(figsize=(15, 6))
    plt.bar(person_image_counts.keys(), person_image_counts.values(), color='skyblue')
    plt.xlabel('Person ID')
    plt.ylabel('Number of Images')
    plt.title('Dataset Distribution')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_PATH, 'dataset_distribution.png')
    plt.savefig(plot_path)
    plt.close(fig)

    summary = (f"Dataset summary:\n- Total number of people: {len(person_groups)}\n"
               f"- Total number of folders: {len(person_folders)}\n"
               f"- Total number of images: {sum(person_image_counts.values())}")
    
    return person_groups, summary, plot_path

def process_image_batch(batch_data):
    """Process a batch of images in parallel"""
    X_batch = []
    y_batch = []
    errors = []
    
    for img_path, label in batch_data:
        try:
            image = cv2.imread(img_path)
            if image is None:
                errors.append(f"Could not read {img_path}")
                continue
                
            # Convert to grayscale and apply enhancements
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image
                
            # Apply enhanced preprocessing for better feature extraction
            image_gray = cv2.equalizeHist(image_gray)
            # CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_gray = clahe.apply(image_gray)
            image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
            image_gray = cv2.resize(image_gray, image_size)
            
            # Convert back to RGB for DenseNet
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            X_batch.append(image_rgb)
            y_batch.append(label)
        except Exception as e:
            errors.append(f"Error processing {img_path}: {str(e)}")
            
    return X_batch, y_batch, errors

def preprocess_dataset(app, person_groups):
    """Preprocess images using parallel processing for speed."""
    global DATASET_PATH, OUTPUT_PATH
    
    # Map person IDs to numerical labels
    person_id_to_index = {pid: idx for idx, pid in enumerate(person_groups.keys())}
    app.queue_message("Preprocessing dataset...")
    
    # Collect all image paths and labels
    all_img_paths = []
    all_labels = []
    
    for person_id, folders in person_groups.items():
        label = person_id_to_index[person_id]
        for folder in folders:
            folder_path = os.path.join(DATASET_PATH, folder)
            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in img_files:
                img_path = os.path.join(folder_path, img_file)
                all_img_paths.append(img_path)
                all_labels.append(label)
    
    # Create batches for parallel processing
    batch_size = 200
    batches = []
    for i in range(0, len(all_img_paths), batch_size):
        end = min(i + batch_size, len(all_img_paths))
        batch_paths = all_img_paths[i:end]
        batch_labels = all_labels[i:end]
        batches.append(list(zip(batch_paths, batch_labels)))
    
    app.queue_message(f"Processing {len(all_img_paths)} images in {len(batches)} batches...")
    
    # Process batches in parallel
    X = []
    y = []
    all_errors = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image_batch, batch) for batch in batches]
        for i, future in enumerate(as_completed(futures)):
            X_batch, y_batch, errors = future.result()
            X.extend(X_batch)
            y.extend(y_batch)
            all_errors.extend(errors)
            
            # Update progress periodically
            if (i + 1) % 5 == 0 or i == len(futures) - 1:
                app.queue_message(f"Processed {i+1}/{len(futures)} batches ({len(X)} images so far)")
    
    if all_errors:
        app.queue_message(f"Encountered {len(all_errors)} errors during processing.")
        
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    app.queue_message(f"Splitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    app.queue_message(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Save a sample of training images to a file
    fig = plt.figure(figsize=(12, 8))
    for i in range(min(9, len(X_train))):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_train[i])
        plt.title(f"Person ID: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    
    sample_path = os.path.join(OUTPUT_PATH, 'sample_images.png')
    plt.savefig(sample_path)
    plt.close(fig)

    # Save preprocessed arrays
    app.queue_message("Saving preprocessed data...")
    np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)
    
    # Save mapping for later use
    np.save(os.path.join(OUTPUT_PATH, 'person_id_mapping.npy'), person_id_to_index)
    
    app.queue_message("Preprocessing completed successfully.")
    
    return X_train, X_test, y_train, y_test, sample_path

def build_densenet_model(num_classes, input_shape=(128, 128, 3), confidence_threshold=0.5):
    """Build a smaller, faster DenseNet model with uncategorized output capability."""
    # Use a smaller version of DenseNet with reduced layers
    base_model = DenseNet121(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze most base layers but not all
    for layer in base_model.layers[:-30]:  # Keep more trainable layers
        layer.trainable = False
        
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Two dense layers for better feature learning
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Reduced dropout for higher confidence
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # Even lower dropout in final layer
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use mixed precision for faster training if on compatible GPU
    if len(physical_devices) > 0:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Store the confidence threshold as a model attribute
    model.confidence_threshold = confidence_threshold
    
    return model, base_model

# Custom callback to report progress
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app
        
    def on_epoch_begin(self, epoch, logs=None):
        self.app.queue_message(f"Starting epoch {epoch+1}...")
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        self.app.queue_message(f"Epoch {epoch+1} completed: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")

def fast_train_densenet(app, X_train, y_train, X_test, y_test, confidence_threshold=0.5):
    """Train a simplified DenseNet model with fewer epochs and optimizations for speed."""
    app.queue_message("Training fast DenseNet model...")
    
    num_classes = len(np.unique(y_train))
    app.queue_message(f"Building model for {num_classes} classes with confidence threshold {confidence_threshold}...")
    model, base_model = build_densenet_model(num_classes, confidence_threshold=confidence_threshold)
    
    model_path = os.path.join(OUTPUT_PATH, 'models', 'best_densenet_model.keras')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0    
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=0
    )
    training_callback = TrainingCallback(app)
    callbacks = [checkpoint, early_stopping, reduce_lr, training_callback]
    
    # Training in two stages for better results while maintaining speed
    app.queue_message("Stage 1: Training top layers...")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,  # Quick initial training
        batch_size=32,
        callbacks=callbacks,
        verbose=0  # Disable verbose output since we're using our custom callback
    )
    
    # Second stage - fine-tune more layers
    app.queue_message("Stage 2: Fine-tuning more layers...")
    # Unfreeze more layers for better feature extraction
    for layer in base_model.layers[-50:]:  # Unfreeze more layers in second stage
        layer.trainable = True
        
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,  # More epochs for fine-tuning
        batch_size=16,  # Smaller batch size for fine-tuning
        callbacks=callbacks,
        verbose=0  # Disable verbose output
    )
    
    # Evaluate on test set
    app.queue_message("Evaluating model...")
    y_pred = model.predict(X_test, batch_size=64, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_conf = np.max(y_pred, axis=1)
    
    # Apply confidence threshold
    y_pred_thresholded = y_pred_classes.copy()
    y_pred_thresholded[y_pred_conf < confidence_threshold] = -1  # -1 indicates "unknown"
    
    # Calculate basic accuracy (without threshold)
    basic_accuracy = np.mean(y_pred_classes == y_test)
    app.queue_message(f"DenseNet Basic Accuracy: {basic_accuracy*100:.2f}%")
    
    # Count how many would be classified as unknown
    unknown_count = np.sum(y_pred_conf < confidence_threshold)
    app.queue_message(f"Samples below confidence threshold ({confidence_threshold}): {unknown_count} ({unknown_count/len(y_test)*100:.2f}%)")
    
    # Create plots
    # Plot training history
    fig1 = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Combine histories
    all_acc = history1.history['accuracy'] + history2.history['accuracy']
    all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    plt.plot(all_acc, label='Train')
    plt.plot(all_val_acc, label='Validation')
    plt.axvline(x=len(history1.history['accuracy'])-0.5, color='r', linestyle='--', label='Stage 2 Start')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    all_loss = history1.history['loss'] + history2.history['loss']
    all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
    plt.plot(all_loss, label='Train Loss')
    plt.plot(all_val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.history['loss'])-0.5, color='r', linestyle='--', label='Stage 2 Start')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    training_plot_path = os.path.join(OUTPUT_PATH, 'training_history.png')
    plt.savefig(training_plot_path)
    plt.close(fig1)
    
    # Plot confidence distribution
    fig2 = plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_conf, bins=20, kde=True)
    plt.axvline(x=confidence_threshold, color='r', linestyle='--', label=f'Threshold ({confidence_threshold})')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    
    confidence_plot_path = os.path.join(OUTPUT_PATH, 'confidence_distribution.png')
    plt.savefig(confidence_plot_path)
    plt.close(fig2)
    
    # Confusion matrix (only for predictions above threshold)
    confusion_plot_path = None
    mask = y_pred_conf >= confidence_threshold
    if np.sum(mask) > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test[mask], y_pred_classes[mask])
        fig3 = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('DenseNet Confusion Matrix (Confident Predictions Only)')
        
        confusion_plot_path = os.path.join(OUTPUT_PATH, 'confusion_matrix.png')
        plt.savefig(confusion_plot_path)
        plt.close(fig3)
    else:
        app.queue_message("No predictions above threshold to create confusion matrix.")
    
    # Add confidence_threshold as an attribute of the model
    model.confidence_threshold = confidence_threshold
    
    app.queue_message(f"Model saved to {model_path}")
    
    return model, basic_accuracy, training_plot_path, confidence_plot_path, confusion_plot_path

def identify_test_image(app, model, test_image_path):
    """Process a test image and use the trained model to predict the person, with unknown detection."""
    image = cv2.imread(test_image_path)
    if image is None:
        app.queue_message("Unable to read image.")
        return None
    
    # Process image with same enhancements as training
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    
    image_gray = cv2.equalizeHist(image_gray)
    # CLAHE for better contrast - same as in training
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
    image_gray = cv2.resize(image_gray, image_size)
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    image_rgb = image_rgb / 255.0
    image_rgb = np.expand_dims(image_rgb, axis=0)
    
    # Get prediction with confidence
    prediction = model.predict(image_rgb, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    # Check if confidence is below threshold
    threshold = getattr(model, 'confidence_threshold', 0.5) * 100
    is_unknown = confidence < threshold
    
    # Create a visualization
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Test Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if is_unknown:
        result_text = f"Result: UNKNOWN\nConfidence too low: {confidence:.2f}%\nThreshold: {threshold:.2f}%"
    else:
        result_text = f"Person ID: {predicted_class}\nConfidence: {confidence:.2f}%"
    
    plt.text(0.5, 0.5, result_text, ha='center', va='center', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    result_path = os.path.join(OUTPUT_PATH, 'identification_result.png')
    plt.savefig(result_path)
    plt.close(fig)
    
    # Log detailed results
    if is_unknown:
        app.queue_message(f"This fingerprint is UNKNOWN (confidence {confidence:.2f}% below threshold {threshold:.2f}%)")
    else:
        app.queue_message(f"Identified as Person ID: {predicted_class}")
        app.queue_message(f"Confidence: {confidence:.2f}%")

    # Show top 3 matches
    top_indices = np.argsort(prediction[0])[::-1][:3]
    app.queue_message("\nTop 3 matches:")
    for i, idx in enumerate(top_indices):
        app.queue_message(f"{i+1}. Person ID: {idx}, Confidence: {prediction[0][idx]*100:.2f}%")
    
    return result_path

def update_threshold_analysis(app, model, X_test, y_test, new_threshold):
    """Analyze the impact of updating the confidence threshold."""
    if model is None:
        return None
        
    # Get predictions
    y_pred = model.predict(X_test, batch_size=64, verbose=0)
    y_pred_conf = np.max(y_pred, axis=1)
    unknown_count = np.sum(y_pred_conf < new_threshold)
    
    app.queue_message(f"With threshold {new_threshold}: {unknown_count} samples ({unknown_count/len(y_test)*100:.2f}%) would be classified as unknown")
    
    # Plot updated confidence distribution
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_conf, bins=20, kde=True)
    plt.axvline(x=new_threshold, color='r', linestyle='--', label=f'New Threshold ({new_threshold})')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    
    threshold_plot_path = os.path.join(OUTPUT_PATH, 'threshold_analysis.png')
    plt.savefig(threshold_plot_path)
    plt.close(fig)
    
    # Update the model's threshold
    model.confidence_threshold = new_threshold
    
    return threshold_plot_path

# ------------------ GUI Application ------------------

class FingerVeinApp:
    def __init__(self, master):
        self.master = master
        master.title("Finger Vein Identification using Fast DenseNet")
        master.geometry("700x600")
        
        # Create main content frame with padding
        main_frame = tk.Frame(master, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section: Control panel
        control_frame = tk.LabelFrame(main_frame, text="Controls", padx=5, pady=5)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Dataset controls
        dataset_frame = tk.Frame(control_frame)
        dataset_frame.pack(fill=tk.X, pady=5)
        
        self.upload_button = tk.Button(dataset_frame, text="Upload Dataset", width=15, command=self.upload_dataset)
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        self.preprocess_button = tk.Button(dataset_frame, text="Preprocess Data", width=15, command=self.preprocess_data)
        self.preprocess_button.pack(side=tk.LEFT, padx=5)
        
        # Training controls
        training_frame = tk.Frame(control_frame)
        training_frame.pack(fill=tk.X, pady=5)
        
        threshold_label = tk.Label(training_frame, text="Confidence Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = tk.Scale(training_frame, from_=0.1, to=0.95, resolution=0.05, 
                                     orient=tk.HORIZONTAL, variable=self.threshold_var, length=200)
        self.threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.train_button = tk.Button(training_frame, text="Train Model", width=15, command=self.train_model)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        # Identification controls
        identify_frame = tk.Frame(control_frame)
        identify_frame.pack(fill=tk.X, pady=5)
        
        self.identify_button = tk.Button(identify_frame, text="Identify Image", width=15, command=self.identify_image)
        self.identify_button.pack(side=tk.LEFT, padx=5)
        
        self.update_threshold_button = tk.Button(identify_frame, text="Update Threshold", width=15, command=self.update_threshold)
        self.update_threshold_button.pack(side=tk.LEFT, padx=5)
        
        self.load_model_button = tk.Button(identify_frame, text="Load Model", width=15, command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5)
        
        # Middle section: Image display
        self.image_frame = tk.LabelFrame(main_frame, text="Visualization", padx=5, pady=5)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.image_label = tk.Label(self.image_frame, text="No image to display")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom section: Log and status
        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Log area
        log_frame = tk.LabelFrame(bottom_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        status_label = tk.Label(status_frame, text="Status:")
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_display = tk.Label(status_frame, textvariable=self.status_var, fg="blue", font=("Arial", 10, "bold"))
        self.status_display.pack(side=tk.LEFT, padx=5)
        
        # Application state variables
        self.person_groups = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.densenet_model = None
        self.model_accuracy = None
        self.training_in_progress = False
        self.processing_in_progress = False
        
        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()
        
        # Start the message queue processing
        self.process_messages()
        
        # Initial log message
        self.log("Finger Vein Identification System initialized")
        self.log("Please upload a dataset to begin")

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()

    def update_status(self, status):
        """Update status display"""
        self.status_var.set(status)
        self.master.update_idletasks()
    
    def queue_message(self, message):
        """Add a message to the queue for thread-safe logging"""
        self.message_queue.put(("log", message))
    
    def process_messages(self):
        """Process messages from the queue"""
        try:
            while not self.message_queue.empty():
                message_type, content = self.message_queue.get_nowait()
                
                if message_type == "log":
                    self.log(content)
                elif message_type == "status":
                    self.update_status(content)
                elif message_type == "image":
                    self.display_image(content)
                elif message_type == "processing_done":
                    self.processing_in_progress = False
                    self.update_status("Ready")
                elif message_type == "training_done":
                    self.training_in_progress = False
                    self.update_status("Ready")
        except queue.Empty:
            pass
        finally:
            # Schedule to run again after 100ms
            self.master.after(100, self.process_messages)

    def display_image(self, image_path, title=None):
        """Display an image in the image frame"""
        if not os.path.exists(image_path):
            self.log(f"Image file not found: {image_path}")
            return
            
        try:
            # Clear previous content
            for widget in self.image_frame.winfo_children():
                widget.destroy()
                
            # Load and resize image
            img = Image.open(image_path)
            img.thumbnail((650, 450))  # Resize while maintaining aspect ratio
            photo = ImageTk.PhotoImage(img)
            
            # Create label to display image
            img_label = tk.Label(self.image_frame, image=photo)
            img_label.image = photo  # Keep a reference to prevent garbage collection
            img_label.pack(fill=tk.BOTH, expand=True)
            
            # Update frame title if provided
            if title:
                self.image_frame.config(text=f"Visualization: {title}")
                
        except Exception as e:
            self.log(f"Error displaying image: {str(e)}")

    def upload_dataset(self):
        """Select dataset folder and explore it"""
        global DATASET_PATH
        
        path = filedialog.askdirectory(title="Select Finger Vein Dataset Folder")
        if not path:
            return
            
        DATASET_PATH = path
        self.update_status("Loading dataset...")
        self.log(f"Dataset path set to: {DATASET_PATH}")
        
        # Use a thread to prevent UI freezing
        def explore_task():
            try:
                self.person_groups, summary, plot_path = explore_dataset()
                
                # Update UI in main thread
                self.master.after(0, lambda: self.log(summary))
                self.master.after(0, lambda: self.display_image(plot_path, "Dataset Distribution"))
                self.master.after(0, lambda: self.update_status("Dataset loaded"))
                
            except Exception as e:
                self.master.after(0, lambda: self.log(f"Error loading dataset: {str(e)}"))
                self.master.after(0, lambda: self.update_status("Error"))
        
        threading.Thread(target=explore_task, daemon=True).start()

    def preprocess_data(self):
        """Preprocess the dataset"""
        if self.person_groups is None:
            messagebox.showerror("Error", "Please upload a dataset first.")
            return
            
        if self.processing_in_progress:
            messagebox.showinfo("Processing", "Preprocessing is already in progress.")
            return
            
        self.processing_in_progress = True
        self.update_status("Preprocessing...")
        
        # Use a thread to prevent UI freezing
        def preprocess_task():
            try:
                self.X_train, self.X_test, self.y_train, self.y_test, sample_path = preprocess_dataset(self, self.person_groups)
                
                # Update UI in main thread
                self.master.after(0, lambda: self.display_image(sample_path, "Sample Images"))
                self.master.after(0, lambda: self.update_status("Ready to train"))
                
            except Exception as e:
                self.master.after(0, lambda: self.log(f"Error during preprocessing: {str(e)}"))
                self.master.after(0, lambda: self.update_status("Error"))
                
            finally:
                self.processing_in_progress = False
        
        threading.Thread(target=preprocess_task, daemon=True).start()

    def train_model(self):
        """Train the DenseNet model"""
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Error", "Please preprocess the dataset first.")
            return
            
        if self.training_in_progress:
            messagebox.showinfo("Training", "Training is already in progress.")
            return
            
        # Get threshold value
        threshold = self.threshold_var.get()
        self.training_in_progress = True
        self.update_status(f"Training model (threshold: {threshold})...")
        
        # Use a thread to prevent UI freezing
        def train_task():
            try:
                self.densenet_model, self.model_accuracy, training_plot, confidence_plot, confusion_plot = fast_train_densenet(
                    self, self.X_train, self.y_train, self.X_test, self.y_test, confidence_threshold=threshold
                )
                
                # Update UI in main thread
                self.master.after(0, lambda: self.display_image(training_plot, "Training History"))
                self.master.after(0, lambda: self.update_status(f"Training completed (Accuracy: {self.model_accuracy*100:.2f}%)"))
                
            except Exception as e:
                self.master.after(0, lambda: self.log(f"Error during training: {str(e)}"))
                self.master.after(0, lambda: self.update_status("Training failed"))
                
            finally:
                self.training_in_progress = False
        
        threading.Thread(target=train_task, daemon=True).start()

    def update_threshold(self):
        """Update the confidence threshold for the model"""
        if self.densenet_model is None:
            messagebox.showerror("Error", "Please train a model first.")
            return
            
        new_threshold = self.threshold_var.get()
        self.update_status(f"Updating threshold to {new_threshold}...")
        
        # Use a thread to prevent UI freezing
        def threshold_task():
            try:
                threshold_plot = update_threshold_analysis(self, self.densenet_model, self.X_test, self.y_test, new_threshold)
                
                # Update UI in main thread
                if threshold_plot:
                    self.master.after(0, lambda: self.display_image(threshold_plot, f"Threshold Analysis ({new_threshold})"))
                
                self.master.after(0, lambda: self.update_status(f"Threshold updated to {new_threshold}"))
                
            except Exception as e:
                self.master.after(0, lambda: self.log(f"Error updating threshold: {str(e)}"))
                self.master.after(0, lambda: self.update_status("Error"))
        
        threading.Thread(target=threshold_task, daemon=True).start()

    def identify_image(self):
        """Identify a person from a test image"""
        if self.densenet_model is None:
            messagebox.showerror("Error", "Please train a model first.")
            return
            
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Finger Vein Image", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
            
        self.update_status("Identifying image...")
        self.log(f"Selected image: {file_path}")
        
        # Use a thread to prevent UI freezing
        def identify_task():
            try:
                result_path = identify_test_image(self, self.densenet_model, file_path)
                
                # Update UI in main thread
                if result_path:
                    self.master.after(0, lambda: self.display_image(result_path, "Identification Result"))
                
                self.master.after(0, lambda: self.update_status("Identification completed"))
                
            except Exception as e:
                self.master.after(0, lambda: self.log(f"Error during identification: {str(e)}"))
                self.master.after(0, lambda: self.update_status("Identification failed"))
        
        threading.Thread(target=identify_task, daemon=True).start()

    def load_model(self):
        """Load a saved model"""
        model_path = os.path.join(OUTPUT_PATH, 'models', 'best_densenet_model.keras')
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "No saved model found.")
            return
            
        try:
            self.update_status("Loading model...")
            self.log(f"Loading model from {model_path}")
            
            # Load the model
            self.densenet_model = load_model(model_path)
            
            # Set confidence threshold
            threshold = self.threshold_var.get()
            self.densenet_model.confidence_threshold = threshold
            
            # Try to load test data if available
            x_test_path = os.path.join(OUTPUT_PATH, 'X_test.npy')
            y_test_path = os.path.join(OUTPUT_PATH, 'y_test.npy')
            
            if os.path.exists(x_test_path) and os.path.exists(y_test_path):
                self.X_test = np.load(x_test_path)
                self.y_test = np.load(y_test_path)
                
            self.update_status("Model loaded")
            self.log("Model loaded successfully")
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            self.update_status("Error")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerVeinApp(root)
    root.mainloop()