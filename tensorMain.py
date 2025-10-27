import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use only GPU 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
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
from tensorflow.keras.mixed_precision import set_global_policy
import psutil
import GPUtil
import socket

# Disable warnings again just in case
warnings.filterwarnings('ignore')

# --- Path Functions ---

def get_base_path():
    computer_name = socket.gethostname()
    if computer_name == "Flippy":
        return "c:/Users/carlo/"
    elif computer_name == "erikjan-desktop":
        return "/media/erikjan/SeagateC25_stora/"
    else:
        return "default/path/"

root_dir = get_base_path()+'cdac Dropbox/Carlos Nunez-Sosa/pedLimb0/'
folders_dir = get_base_path()+'cdac Dropbox/Carlos Nunez-Sosa/pedLimb0/tfrecords/'

# --- Model Definition (No changes needed here) ---

def build_unet_model_ayt(input_shape=(134400, 2), alpha=1.67):
    # ... (Model Definition remains the same) ...
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

# --- F1 Metric (Custom is left in but not used in compile) ---

def f1_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    

    tp = K.sum(y_true * y_pred) # <-- Note: Removed axis=0 for proper summation
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# --- Weighted Focal Loss (CRITICAL: Needs low learning rate for stability) ---

def create_weighted_focal_loss(class_weight_dict, gamma=2.0):
    def weighted_focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Fix Shape Mismatch: y_true is (batch, 2100), y_pred is (batch, 2100, 1)
        # tf.expand_dims is crucial for element-wise multiplication below
        y_true = tf.expand_dims(y_true, axis=-1)

        # Numerical Stability: Clip predictions to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Get the weights for class 0 (background) and class 1 (movement)
        weight_for_0 = class_weight_dict.get(0, 1.0)
        weight_for_1 = class_weight_dict.get(1, 1.0)

        # Focal Loss components
        # pt for class 1 (where y_true is 1) is y_pred
        pt_1 = y_pred 
        # pt for class 0 (where y_true is 0) is (1 - y_pred)
        pt_0 = 1. - y_pred

        # Loss calculation (y_true acts as the mask/indicator for class 1)
        loss_for_1 = -weight_for_1 * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
        loss_for_0 = -weight_for_0 * tf.math.pow(1. - pt_0, gamma) * tf.math.log(pt_0)
        
        # Combine the losses: y_true selects loss_for_1, (1-y_true) selects loss_for_0
        loss = y_true * loss_for_1 + (1. - y_true) * loss_for_0

        return tf.reduce_mean(loss)

    return weighted_focal_loss

# --- Data Pipeline Functions (Assuming parse_tfrecord is correct) ---

def create_dataset(tfrecord_files, batch_size=64, shuffle_buffer_size=100, prefetch_buffer_size=1):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE) # Use AUTOTUNE
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset

def parse_tfrecord(example_proto):
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([2100], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    signals = tf.reshape(signals, (134400, 2))

    annotations = tf.cast(parsed_example['annotations'], tf.float32)
    return signals, annotations

def compute_class_weights(train_dataset):
    """
    Calculates class weights and total steps for the Keras fit function.
    """
    print("Computing class weights efficiently...")
    class_counts = {0: 0, 1: 0}
    total_batches = 0 
    
    # Iterate through the dataset once to count labels
    for _, annotations in train_dataset:
        total_batches += 1
        labels = annotations.numpy().flatten()
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if label in class_counts:
                class_counts[label] += count
    
    if class_counts[0] == 0 or class_counts[1] == 0:
        print("Warning: One class has zero samples in the dataset.")
        return {0: 1.0, 1: 1.0}, total_batches 

    # Create dummy array for sklearn's utility to get balanced weights
    class_labels = np.array(list(class_counts.keys()))
    y_dummy = np.concatenate([np.full(class_counts[cls], cls) for cls in class_labels])
    
    balanced_weights = compute_class_weight(
        "balanced",
        classes=class_labels,
        y=y_dummy
    )
    
    class_weight_dict = dict(zip(class_counts.keys(), balanced_weights))

    print("Computed Class Weights:", class_weight_dict)
    print(f"Total training batches found: {total_batches}")
    return class_weight_dict, total_batches 

def squeeze_output_metric(y_true, y_pred):
    # This function is run inside the TF graph and handles the reshape
    y_pred_squeezed = tf.squeeze(y_pred, axis=-1)
    
    # We can now use the FBetaScore class directly
    # Note: Keras metrics are stateful, so we must instantiate the metric here
    # Use your custom f1_m which handles the squeeze internally:
    return f1_m(y_true, y_pred_squeezed) # <--- Re-use your custom f1_m function

# --- End Data Pipeline Functions ---


# --- Strategy and Data Loading ---

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

trainFold = folders_dir + 'train'
valFold = folders_dir + 'val'
testFold = folders_dir + 'test'

# ... (Folder and TFRecord file list generation remains the same) ...
train_folders = [f for f in os.listdir(trainFold) if os.path.isdir(os.path.join(trainFold, f))]
val_folders = [f for f in os.listdir(valFold) if os.path.isdir(os.path.join(valFold, f))]
test_folders = [f for f in os.listdir(testFold) if os.path.isdir(os.path.join(testFold, f))]

print(len(train_folders), "training folders found.")

train_tfrecords = [os.path.join(trainFold, f, f"{f}.tfrecord") for f in train_folders]
train_dataset = create_dataset(train_tfrecords, batch_size=64)
for signals, annotations in train_dataset.take(1):
    print("Train data shape:", signals.shape, annotations.shape)
   
val_tfrecords = [os.path.join(valFold, f, f"{f}.tfrecord") for f in val_folders]
val_dataset = create_dataset(val_tfrecords, batch_size=64)
for signals, annotations in val_dataset.take(1):
    print("Validation data shape:", signals.shape, annotations.shape)

test_tfrecords = [os.path.join(testFold, f, f"{f}.tfrecord") for f in test_folders]
test_dataset = create_dataset(test_tfrecords, batch_size=64)
for signals, annotations in test_dataset.take(1):
   print("Test data shape:", signals.shape, annotations.shape)




# --- Compilation and Training ---

class_weight_dict, train_steps = compute_class_weights(train_dataset) 

# --- START STRATEGY SCOPE ---
with strategy.scope():
    model = build_unet_model_ayt(input_shape=(134400, 2))
    
    # CRITICAL FIX: Lower the learning rate for Focal Loss stability
    optimizer = keras.optimizers.Adam(learning_rate=0.00005) # Reduced from 0.0001
    
    focal_loss_fn = create_weighted_focal_loss(class_weight_dict, gamma=2.0)
    
    # Use the Keras F1 metric
    # Compile INSIDE the scope
    model.compile(optimizer=optimizer, 
                  loss=focal_loss_fn, 
                  metrics=['accuracy', squeeze_output_metric])
# --- END STRATEGY SCOPE ---

print("Model Compiled with Weighted Focal Loss under MirroredStrategy")

# Early stopping now monitors the new 'val_f1_score' metric
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=10, mode='max', restore_best_weights=True)

# CRITICAL FIX: Add steps_per_epoch and .repeat()
model.fit(
    train_dataset.repeat(),        # Must repeat so TF doesn't exhaust the dataset after one pass
    validation_data=val_dataset,
    epochs=1000,
    callbacks=[early_stopping],
    steps_per_epoch=train_steps,   # FIX: Explicitly tells TF how many batches are in an epoch
    validation_steps=len(val_tfrecords), # Good practice: Tell TF how many batches are in val_set
    verbose=1
)


################################# Save Model ##################################

model.save("trained_model_usleep_new.h5")