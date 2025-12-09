import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings
import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings

# --- 1. CONFIGURATION ---
# --- Point this to your FAST SSD drive ---
tfFOLDERS = '/mnt/SeagateC25_stora/pedLimbDetectCNS/tfrecords/' 
TEST_TFRECORD_DIR = os.path.join(tfFOLDERS, "test")
MODEL_PATH = "best_model_custom_loop.keras" # Point to your .keras file
BATCH_SIZE = 32

# --- Constants (Must match training) ---
# It's safer to redefine them here than import, to avoid circular dependency errors
SEGMENT_COUNT = 35
SEGMENT_LEN_SEC = 30
SF_TARGET = 128
ANOT_TARGET_FREQ = 2
NUM_CHANNELS = 2
SIGNAL_SAMPLES = 134400 
ANOT_SAMPLES = 2100 

# --- 2. Helper Functions (Needed for Data Loading) ---

def parse_tfrecord(example_proto):
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES))
    signals = tf.transpose(signals)
    annotations = tf.cast(parsed_example['annotations'], tf.int32)
    return signals, annotations

def create_dataset(tfrecord_files, batch_size=64, is_training=False):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 3. Custom Loss Definition (Needed for Loading) ---
# We must define this EXACTLY as it was in the training script
import tensorflow.keras.backend as K
def weighted_focal_loss(pos_weight, neg_weight, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        loss_pos = -pos_weight * y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred)
        loss_neg = -neg_weight * (1 - y_true) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss_pos + loss_neg)
    return loss

def main():
    print(f"--- 1. Finding Test Files ---")
    test_tfrecords = glob.glob(os.path.join(TEST_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    if not test_tfrecords:
        raise FileNotFoundError(f"No test files found at {TEST_TFRECORD_DIR}")
    print(f"Found {len(test_tfrecords)} test files.")

    # --- 2. Load Test Dataset ---
    print(f"--- 2. Loading Test Dataset (Batch Size: {BATCH_SIZE}) ---")
    test_dataset = create_dataset(test_tfrecords, batch_size=BATCH_SIZE, is_training=False)

    # --- 3. Load Model ---
    print(f"--- 3. Loading Model from {MODEL_PATH} ---")
    
    # CRITICAL STEP:
    # When loading a model with a custom loss that requires arguments (like pos_weight),
    # we often have to load it without the optimizer/loss first, or provide a 
    # dummy version of the loss function.
    
    # Strategy: Register the inner 'loss' function name if Keras saved it that way,
    # OR simpler: Load with compile=False.
    # Since we are only evaluating (predicting), we do NOT need the optimizer or loss function!
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully (compiled=False).")
    except Exception as e:
        print(f"\n--- MODEL LOADING FAILED ---")
        print(f"Error: {e}")
        return

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
    
    f1_binary = f1_score(all_true_labels, all_predictions, pos_label=1, average='binary', zero_division=0)
    precision_binary = precision_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
    recall_binary = recall_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)

    print("\n--- Metrics for 'Movement' (Class 1) ---")
    print(f"F1 Score (Class 1):    {f1_binary:.4f}")
    print(f"Precision (Class 1): {precision_binary:.4f}")
    print(f"Recall (Class 1):    {recall_binary:.4f}")

    print("\n--- Full Classification Report ---")
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=['Class 0 (No Move)', 'Class 1 (Movement)'], 
        zero_division=0
    )
    print(report)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
    main()