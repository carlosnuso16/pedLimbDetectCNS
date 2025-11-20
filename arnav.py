
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
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
from sklearn.metrics import f1_score, confusion_matrix
import logging
tf.get_logger().setLevel(logging.ERROR)
from tqdm import tqdm
import tensorflow.keras.backend as K




def build_usleep_model_ayt(input_shape=(134400, 10), alpha=1.67):

    def encoder_block(x, filters, kernel_size=9):
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)  # Apply dropout only if specified
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
       
        # Crop or pad the residual connection to match x's shape
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
   
###############################    TFRecord parsing   #################

def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example into signals and annotations.

    Args:
        example_proto: Serialized TFRecord example.

    Returns:
        signals: Tensor of shape (134400, 8).
        annotations: Tensor of shape (35,).
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([2100], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode signals
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    signals = tf.reshape(signals, (134400, 10))

    # Extract annotations
    annotations = tf.cast(parsed_example['annotations'], tf.int32)
    

    return signals, annotations



def create_dataset(tfrecord_files, batch_size=64, shuffle_buffer_size=100, prefetch_buffer_size=tf.data.AUTOTUNE):
    """
    Create a tf.data.Dataset pipeline for TFRecords.

    Args:
        tfrecord_files (list): List of TFRecord file paths.
        batch_size (int): Batch size for training.
        shuffle_buffer_size (int): Buffer size for shuffling.
        prefetch_buffer_size: Buffer size for prefetching.

    Returns:
        A tf.data.Dataset object.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)  # Load TFRecords
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)  # Parse each record
    dataset = dataset.shuffle(shuffle_buffer_size)  # Shuffle dataset
    dataset = dataset.batch(batch_size)  # Batch the data
    dataset = dataset.prefetch(prefetch_buffer_size)  # Prefetch for performance
    return dataset
   
   
def update_confusion(conf_mat, y_true, y_pred, num_classes=2):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    if conf_mat is None:
        return cm
    else:
        return conf_mat + cm
       
def f1_confusion(conf):
    # conf: shape (2, 2) for binary classification
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)
    macro_f1 = np.mean(f1)
    return macro_f1

def arousal_f1_from_confusion(conf):
    tp = conf[1, 1]
    fp = conf[0, 1]
    fn = conf[1, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

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


train_folders = [os.path.join('/media/cdac/lachesis/train_orig', d) for d in os.listdir('/media/cdac/lachesis/train_orig')]
print(len(train_folders))
train_tfrecords = [os.path.join(f, os.path.basename(f) + '.tfrecord') for f in train_folders]
train_dataset = create_dataset(train_tfrecords, batch_size=64,shuffle_buffer_size=10)  

val_folders = [os.path.join('/media/cdac/311f8483-7bbc-4353-a171-8eb9b61bf683/val_orig', d) for d in os.listdir('/media/cdac/311f8483-7bbc-4353-a171-8eb9b61bf683/val_orig')]
val_tfrecords = [os.path.join(f, os.path.basename(f) + '.tfrecord') for f in val_folders]  
val_dataset = create_dataset(val_tfrecords, batch_size=64,shuffle_buffer_size=10)

model = build_usleep_model_ayt(input_shape=(134400, 10))
print(model.summary())
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
class_weights = {0: 0.5044633711286198, 1: 56.51147491342541}
loss_fn = weighted_focal_loss(pos_weight=class_weights[1], neg_weight=class_weights[0], gamma=2.0)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


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

best_val_f1 = 0
patience = 20
wait = 0
num_epochs = 1000

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    batch_losses, batch_accuracies = [], []
    train_conf = None

    progbar = tqdm(train_dataset, desc="Training", leave=False)
    for x_batch, y_batch in progbar:
        loss, acc = train_step(x_batch, y_batch)
        batch_losses.append(loss.numpy())
        batch_accuracies.append(acc.numpy())

    epoch_loss = np.mean(batch_losses)
    epoch_acc = np.mean(batch_accuracies)
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    val_conf = None
    for i, (x_batch, y_batch) in enumerate(val_dataset):
        preds = eval_step(x_batch)
        y_pred = (preds.numpy().squeeze(-1) >= 0.5).astype(int).flatten()
        val_conf = update_confusion(val_conf, y_batch.numpy().flatten(), y_pred)
        del x_batch, y_batch, preds  # Free memory

    val_f1_arousal = arousal_f1_from_confusion(val_conf)
    print(f"Val F1: {val_f1_arousal:.4f}")

    if val_f1_arousal > best_val_f1:
        best_val_f1 = val_f1_arousal
        wait = 0
        model.save("trained_WFL_arousal.keras")
        print("Best model saved.")
    else:
        wait += 1
        print(f"No improvement. Patience: {wait}/{patience}")
        if wait >= patience:
            print("Early stopping triggered.")
            break

    tf.keras.backend.clear_session()


