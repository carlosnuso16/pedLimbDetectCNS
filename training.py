#conda activate tf_pip_fix

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_TRT_DISABLE"] = "1"   # avoid TensorRT warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import glob
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers
from scipy.signal import iirnotch, butter, filtfilt, hilbert, resample
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K
import mne
from sklearn.utils.class_weight import compute_class_weight
import warnings
from sklearn.metrics import f1_score, confusion_matrix
import logging
tf.get_logger().setLevel(logging.ERROR)
from tqdm import tqdm
import tensorflow.keras.backend as K

# --- [NEW] GPU Check ---
print("--- [INFO] Checking GPU visibility...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"--- [INFO] TensorFlow can see the following GPU(s):")
    for gpu in gpus:
        print(f"   {gpu.name}")
else:
    print("*** [ERROR] TensorFlow cannot see any GPU! Check CUDA_VISIBLE_DEVICES. ***")
# --- [END NEW] ---

# --- 1. Constants (from your old script) ---
SEGMENT_COUNT = 35
SEGMENT_LEN_SEC = 30
SF_TARGET = 128
ANOT_TARGET_FREQ = 2
NUM_CHANNELS = 2 # Rat and Lat

SIGNAL_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * SF_TARGET # 134400
ANOT_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * ANOT_TARGET_FREQ # 2100

# --- 2. Model Architecture ---
def build_usleep_model_ayt(input_shape=(134400, NUM_CHANNELS), alpha=1.67):
    """
    Builds the 1D U-Net model.
    (This is your mentor's model, unchanged)
    """
    def encoder_block(x, filters, kernel_size=9):
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        res = x
        x = layers.ZeroPadding1D((0, 1))(x) if x.shape[1] % 2 != 0 else x
        x = layers.MaxPooling1D(2)(x)
        return x, res

    def decoder_block(x, res, filters, kernel_size=9):
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
       
        diff = res.shape[1] - x.shape[1]
        if diff > 0:
            res = layers.Cropping1D((diff // 2, diff - diff // 2))(res)
        elif diff < 0:
            x = layers.Cropping1D((-diff // 2, -diff - (-diff // 2)))(x)
       
        x = layers.Concatenate()([x, res])
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        return x

    # --- FIX 1: Hard-coded params (set to None) ---
    l2_lambda = None
    dropout_rate = None
    inputs = keras.Input(shape=input_shape)
    x = inputs

    encoder_residuals = []
    filter_sizes = np.array([6, 9, 11, 15, 20, 28, 40, 55, 77, 108, 152, 214])

    for filters in filter_sizes:
        x, res = encoder_block(x, filters)
        encoder_residuals.append(res)

    x = layers.Conv1D(int(306 * np.sqrt(alpha)), 9, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None)(x)
    x = layers.ELU()(x)
    x = layers.BatchNormalization()(x)

    for res, filters in zip(reversed(encoder_residuals), reversed(filter_sizes)):
        x = decoder_block(x, res, filters)

    x = layers.Conv1D(6, 1, padding='same', activation='tanh')(x)
    x = layers.AveragePooling1D(pool_size=64)(x)
    x = layers.Conv1D(5, 1, padding='same', activation='elu')(x)
    outputs = layers.Conv1D(1, 1, padding='same', activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model
   
# --- 3. TFRecord Parsing & Dataset Creation ---

def parse_tfrecord(example_proto):
    """
    Parses a single TFRecord example.
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    
    # --- FIX 2: Use constants instead of hard-coded numbers ---
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES))
    
    signals = tf.transpose(signals) # Transpose to (134400, 2)
    annotations = tf.cast(parsed_example['annotations'], tf.int32)
    return signals, annotations

def create_dataset(tfrecord_files, batch_size=64, shuffle_buffer_size=100, prefetch_buffer_size=tf.data.AUTOTUNE, is_training=False):
    """
    Create a tf.data.Dataset pipeline for TFRecords.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(shuffle_buffer_size)
        # --- FIX 3: Added .repeat() ---
        # A custom loop *must* have .repeat() on an infinite dataset,
        # otherwise it will only train for one epoch.
        dataset = dataset.repeat()
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

# --- 4. Helper Functions (Loss, F1, Class Weights) ---
   
def update_confusion(conf_mat, y_true, y_pred, num_classes=2):
    """(This is your mentor's function, unchanged)"""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    if conf_mat is None:
        return cm
    else:
        return conf_mat + cm
       
def limb_f1_from_confusion(conf):
    """
    (This is your mentor's function, unchanged)
    It correctly calculates F1 for *only* Class 1. This is good.
    """
    tp = conf[1, 1]
    fp = conf[0, 1]
    fn = conf[1, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def weighted_focal_loss(pos_weight, neg_weight, gamma=2.0):
    """(This is your mentor's function, unchanged)"""
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

# --- FIX 4: Copied this function from your old script ---
def compute_class_weights(tfrecord_files):
    """
    Calculates class weights by counting records.
    We run this *once* at the beginning.
    """
    print("Calculating class weights... This may take a moment.")
    
    # Create a fast, non-batched dataset just for counting
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    all_labels = []
    # Loop over all 17.5-min records
    for _, annotations in tqdm(dataset, desc="Counting labels"):
        all_labels.extend(annotations.numpy().flatten())

    class_labels = np.unique(all_labels)
    class_weights = compute_class_weight("balanced", classes=class_labels, y=all_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
   
    print("Computed Class Weights:", class_weight_dict)
    return class_weight_dict

# --- 5. Main Training Execution ---

# directories containing your .tfrecord files
# --- IMPORTANT: Make sure this points to your SSD ---
tfFOLDERS = '/mnt/SeagateC25_stora/pedLimbDetectCNS/tfrecords/'
TRAIN_TFRECORD_DIR = os.path.join(tfFOLDERS, "train")
VAL_TFRECORD_DIR = os.path.join(tfFOLDERS, "val")
TEST_TFRECORD_DIR = os.path.join(tfFOLDERS, "test")

BATCH_SIZE = 32 # Start with 16, you can increase later if it doesn't crash
NUM_EPOCHS = 1000 # Set high, EarlyStopping will handle it
PATIENCE = 20

print("--- 1. Finding TFRecord Files ---")
train_tfrecords = glob.glob(os.path.join(TRAIN_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
val_tfrecords = glob.glob(os.path.join(VAL_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
test_tfrecords = glob.glob(os.path.join(TEST_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    
print(f"Found {len(train_tfrecords)} training TFRecord files.")
print(f"Found {len(val_tfrecords)} validation TFRecord files.")
print(f"Found {len(test_tfrecords)} test TFRecord files.")

# --- 2. Calculate TRUE steps per epoch ---
# We use the method from count_my_records.py to get the *real* number
# of records, not just the number of files.
print("Counting total records...")
num_train_examples = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecords))
num_val_examples = sum(1 for _ in tf.data.TFRecordDataset(val_tfrecords))

# Use ceiling division to round up
STEPS_PER_EPOCH = (num_train_examples + BATCH_SIZE - 1) // BATCH_SIZE
VALIDATION_STEPS = (num_val_examples + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Total Train Records: {num_train_examples} -> Steps per Epoch: {STEPS_PER_EPOCH}")
print(f"Total Val Records:   {num_val_examples} -> Validation Steps: {VALIDATION_STEPS}")


print("\n--- 3. Creating tf.data.Dataset Pipelines ---")
train_dataset = create_dataset(train_tfrecords, batch_size=BATCH_SIZE, is_training=True)
val_dataset = create_dataset(val_tfrecords, batch_size=BATCH_SIZE, is_training=False)
test_dataset = create_dataset(test_tfrecords, batch_size=BATCH_SIZE, is_training=False)

# --- 4. Compute Class Weights ---
# (This runs a loop over the data, which is fine)
# class_weights = compute_class_weights(train_tfrecords) #temp comment out
class_weights = {0: 0.5018162877934619, 1: 138.14338498553548}

# --- 5. Build, Compile, and Define Steps ---
print("\n--- 5. Building and Compiling Model ---")
model = build_usleep_model_ayt(input_shape=(SIGNAL_SAMPLES, NUM_CHANNELS))
# print(model.summary())
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

loss_fn = weighted_focal_loss(
    pos_weight=class_weights[1], 
    neg_weight=class_weights[0], 
    gamma=2.0
)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
print("Model compiled.")

# --- 6. Define Custom Train/Eval Steps ---
print("\n--- 6. Defining custom train/eval @tf.function steps ---") # <-- ADDED PRINT

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = tf.keras.metrics.binary_accuracy(y, pred)
    return loss, tf.reduce_mean(acc)

@tf.function
def eval_step(x):
    return model(x, training=False)

# --- 7. Start Custom Training Loop ---
best_val_f1 = 0
wait = 0

print(f"\n--- 7. Starting Training ---")
print(f"   Epochs: {NUM_EPOCHS} (with Patience: {PATIENCE})")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Train Steps: {STEPS_PER_EPOCH}")
print(f"   Val Steps: {VALIDATION_STEPS}")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    batch_losses, batch_accuracies = [], []

    # --- Training Loop ---
    progbar = tqdm(train_dataset, desc="Training", total=STEPS_PER_EPOCH, leave=False)
    for i, (x_batch, y_batch) in enumerate(progbar):
        # --- FIX 5: Added manual 'break' ---
        # We must manually stop the loop when we hit our quota
        if i >= STEPS_PER_EPOCH:
            break
            
        loss, acc = train_step(x_batch, y_batch)
        batch_losses.append(loss.numpy())
        batch_accuracies.append(acc.numpy())
        
        if i % 25 == 0: # Print a quick update
             progbar.set_postfix_str(f"Loss: {loss.numpy():.4f}")

    epoch_loss = np.mean(batch_losses)
    epoch_acc = np.mean(batch_accuracies)
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # --- Validation Loop ---
    val_conf = None
    val_progbar = tqdm(val_dataset, desc="Validating", total=VALIDATION_STEPS, leave=False)
    for i, (x_batch, y_batch) in enumerate(val_progbar):
        # --- FIX 6: Added manual 'break' ---
        if i >= VALIDATION_STEPS:
            break
            
        preds = eval_step(x_batch)
        y_pred = (preds.numpy().squeeze(-1) >= 0.5).astype(int).flatten()
        val_conf = update_confusion(val_conf, y_batch.numpy().flatten(), y_pred)
        del x_batch, y_batch, preds

    val_f1_limb = limb_f1_from_confusion(val_conf)
    print(f"Val F1: {val_f1_limb:.4f}")

    # --- Early Stopping Logic ---
    if val_f1_limb > best_val_f1:
        best_val_f1 = val_f1_limb
        wait = 0
        model.save("best_model_custom_loop.keras") # Saved with a new name
        print("Best model saved.")
    else:
        wait += 1
        print(f"No improvement. Patience: {wait}/{PATIENCE}")
        if wait >= PATIENCE:
            print("Early stopping triggered.")
            break
            
    # Clear session to prevent memory leaks in a long loop
    tf.keras.backend.clear_session()