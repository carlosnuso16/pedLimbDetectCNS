import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings
import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings

# --- 1. Import functions and constants from your training script ---
# This assumes 'evaluate.py' is in the same folder as 'train_model_stable.py'
try:
    from tensorMain import (
        parse_tfrecord, 
        create_dataset, 
        weighted_focal_loss,
        ANOT_SAMPLES,
        SIGNAL_SAMPLES,
        NUM_CHANNELS
    )
except ImportError:
    print("Error: Could not import from 'tensorMain.py'.")
    print("Please make sure 'evaluate.py' is in the same directory.")
    exit()

# --- 2. Configuration ---
# --- Point this to your FAST SSD drive ---
tfFOLDERS = '/mnt/SeagateC25_stora/pedLimbDetectCNS/tfrecords/' 
TEST_TFRECORD_DIR = os.path.join(tfFOLDERS, "test")
MODEL_PATH = "firstModel.h5" # The model saved by your F1Callback
BATCH_SIZE = 32 # Can be larger for evaluation

def main():
    print(f"--- 1. Finding Test Files ---")
    test_tfrecords = glob.glob(os.path.join(TEST_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    if not test_tfrecords:
        raise FileNotFoundError(f"No test files found at {TEST_TFRECORD_DIR}")
    print(f"Found {len(test_tfrecords)} test files.")

    # --- 2. Load Test Dataset ---
    print(f"--- 2. Loading Test Dataset (Batch Size: {BATCH_SIZE}) ---")
    # is_training=False ensures no shuffling and no repeating
    test_dataset = create_dataset(test_tfrecords, batch_size=BATCH_SIZE, is_training=False)

    # --- 3. Load Model ---
    print(f"--- 3. Loading Model from {MODEL_PATH} ---")
    # We must provide the custom loss function to load the model
    # The weights don't matter for loading, so we can use a dummy dict
    dummy_weights = {0: 1.0, 1: 1.0}
    custom_objects = {'loss_fn': weighted_focal_loss(dummy_weights)}
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception as e:
        print(f"\n--- MODEL LOADING FAILED ---")
        print(f"Error: {e}")
        print("\nThis often happens if the 'weighted_focal_loss' function name in your")
        print("training script doesn't match what's saved in the model.")
        print("Please ensure your 'train_model_stable.py' is up to date.")
        return

    print("Model loaded successfully.")

    # --- 4. Run Evaluation ---
    print(f"--- 4. Running predictions on {len(test_tfrecords)} test files... ---")
    all_true_labels = []
    all_predictions = []

    # Loop through the entire test set
    for signals, annotations in tqdm(test_dataset):
        # 1. Get model predictions (probabilities)
        preds_prob = model.predict(signals, verbose=0)
        
        # 2. Convert probabilities to binary 0 or 1
        preds_binary = (preds_prob > 0.5).astype(int)
        
        # 3. Store the results
        all_predictions.extend(preds_binary.flatten())
        all_true_labels.extend(annotations.numpy().flatten())

    print("Evaluation complete.")

    # --- 5. Calculate and Print Metrics ---
    print("\n--- 5. FINAL TEST METRICS ---")
    
    # Calculate key metrics for the "Movement" class (pos_label=1)
    # This is the most important metric for you
    f1_binary = f1_score(all_true_labels, all_predictions, pos_label=1, average='binary', zero_division=0)
    precision_binary = precision_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
    recall_binary = recall_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)

    print("\n--- Metrics for 'Movement' (Class 1) ---")
    print(f"F1 Score (Class 1):    {f1_binary:.4f}")
    print(f"Precision (Class 1): {precision_binary:.4f}")
    print(f"Recall (Class 1):    {recall_binary:.4f}")

    print("\n--- Full Classification Report ---")
    # This provides a detailed breakdown for both classes
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=['Class 0 (No Move)', 'Class 1 (Movement)'], 
        zero_division=0
    )
    print(report)

if __name__ == "__main__":
    # Suppress TensorFlow UserWarnings about data running out
    warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
    main()