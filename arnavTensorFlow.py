#Computed Class Weights: {0: 0.5044633711286198, 1: 56.51147491342541}

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use only GPU 1
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
import GPUtil  # Install via `pip install gputil`

# Enable logging of device placement
#tf.debugging.set_log_device_placement(True)

fs = 128

def compute_class_weights(train_dataset):
    """Calculate class weights based on training data distribution."""
    all_labels = []
   
    # Collect all training labels
    for _, annotations in train_dataset:
        all_labels.extend(annotations.numpy().flatten())  # Convert to list

    # Compute class weights
    class_labels = np.unique(all_labels)
    class_weights = compute_class_weight("balanced", classes=class_labels, y=all_labels)

    # Convert to dictionary format for easy indexing
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
   
    print("Computed Class Weights:", class_weight_dict)
    return class_weight_dict

def weighted_focal_loss(class_weight_dict, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true_int = tf.cast(y_true, 'int32')
       
        # Gather class weights
        weights = tf.gather(tf.constant([class_weight_dict[i] for i in sorted(class_weight_dict.keys())], dtype=tf.float32), y_true_int)

        # Compute focal loss
        focal_loss = -y_true * tf.math.log(y_pred) * (1 - y_pred) ** gamma - (1 - y_true) * tf.math.log(1 - y_pred) * (y_pred ** gamma)
        weighted_loss = weights * focal_loss
        return tf.reduce_mean(weighted_loss)

    return loss_fn

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Computes Focal Loss to address class imbalance.

    Args:
        gamma (float): Focusing parameter (default: 2.0).
                       Higher values focus more on difficult examples.
        alpha (float): Balancing parameter (default: 0.25).
                       Helps balance the impact of positive and negative classes.

    Returns:
        A loss function to use in model compilation.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)

        # Prevent log(0) issues
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Compute focal loss
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # pt = p if true class is 1, (1 - p) otherwise
        loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt)

        return tf.reduce_mean(loss)

    return loss_fn


####################################          F1       #############################################
def f1_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')  # Ensure float32
    y_pred = K.cast(y_pred, 'float32')  # Ensure float32
    y_pred = K.squeeze(y_pred, axis=-1)  # Removes last dimension if [64, 2100, 1] � [64, 2100]

    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())  # Avoid division by zero
    recall = tp / (tp + fn + K.epsilon())

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
     
class F1ScoreCallback(Callback):
    def __init__(self, validation_data, train_data):
        super().__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        self.best_f1 = 0.0  # Track best F1 score

    def compute_f1(self, dataset):
        """Helper function to compute F1 score for any dataset"""
        predictions = []
        true_labels = []
   
        for signals, annotations in dataset:
            preds = self.model.predict(signals, verbose=0)
            preds = (preds > 0.5).astype(int)  # Convert to binary labels
            predictions.extend(preds.flatten())
            true_labels.extend(annotations.numpy().flatten())

        return f1_score(true_labels, predictions, average="weighted")  # Weighted F1 for imbalanced classes

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Compute Training F1
        if self.train_data:
            train_f1 = self.compute_f1(self.train_data)
            logs["train_f1_score"] = train_f1
            print(f"\nEpoch {epoch+1}: train_f1_score = {train_f1:.4f}")

        # Compute Validation F1
        val_f1 = self.compute_f1(self.validation_data)
        logs["val_f1_score"] = val_f1
        print(f"Epoch {epoch+1}: val_f1_score = {val_f1:.4f}")

        # Save best model based on F1 score
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.model.save("best_model_f1.h5")

           
#####################          Preprocessing       ###############


def robust_scaling(signal):
    """Apply robust scaling to achieve zero mean and unit variance."""
    median = np.median(signal, axis=0)
    iqr = np.percentile(signal, 75, axis=0) - np.percentile(signal, 25, axis=0)
    if iqr==0:
       iqr=1
    signal = (signal - median) / iqr
    signal = np.clip(signal, -20, 20)
    return signal

def preprocess_signals(signal, fs, signal_name=None):
    """Apply notch filtering at 60 Hz and a high-pass filter at 10 Hz."""

    # Ensure signal is 2D (n_channels, n_times) for MNE
    signal = np.atleast_2d(signal).astype(np.float64)

    # Define channel types explicitly if needed
    ch_types = {
        "lat": "emg",
        "rat": "emg"
    }
   
    ch_type = ch_types.get(signal_name, "misc")  # Default to 'misc'
    ch_names = [signal_name] if signal_name else ["unknown"]

    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=[ch_type])

    # Create RawArray in MNE
    raw = mne.io.RawArray(signal.squeeze()[np.newaxis, :], info)

    print(f"Processing {signal_name} | Assigned Type: {ch_type}")

    # Apply filters
    raw.notch_filter(freqs=60, picks=[0], fir_design="firwin")   # 60 Hz notch
    raw.filter(l_freq=10.0, h_freq=None, picks=[0], fir_design="firwin")  # High-pass at 10 Hz

    # Extract processed signal
    processed_signal = raw.get_data().squeeze()

    return processed_signal

   

def resample_annotations_to_2Hz(annotations, fs_orig=200):
    """
    Downsample annotations from 200Hz to 2Hz.
   
    - Takes the maximum value over every 100-sample window.
    - If any 1 is present in the window, it assigns 1; otherwise, it assigns 0.
   
    Parameters:
        annotations (np.array): Original annotation array at 200Hz.
        fs_orig (int): Original sampling frequency (default: 200Hz).
   
    Returns:
        np.array: Downsampled annotation array at 2Hz.
    """
    assert fs_orig == 200, "Expected original sampling frequency of 200Hz."

    # Compute new size at 2Hz
    num_samples = len(annotations) // 100  # Since 200Hz � 2Hz

    # Apply downsampling by checking 100-sample blocks
    downsampled_annotations = np.array([
        1 if np.any(annotations[i * 100: (i + 1) * 100] == 1) else 0
        for i in range(num_samples)
    ], dtype=np.int32)

    return downsampled_annotations
       
def resample_signal(signal, fs_orig=200, fs_target=128):
    """
    Resample the signal from fs_orig (200 Hz) to fs_target (128 Hz).
    Uses interpolation to avoid aliasing effects.
   
    Parameters:
        signal (numpy array): The signal to be resampled.
        fs_orig (int): Original sampling rate (default 200 Hz).
        fs_target (int): Target sampling rate (default 128 Hz).
   
    Returns:
        numpy array: Resampled signal.
    """
    num_samples = int(signal.shape[0] * (fs_target / fs_orig))  # Compute new number of samples
    return resample(signal, num_samples, axis=0)


def build_usleep_model_ayt(input_shape=(134400, 2), alpha=1.67):

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

#####################      U-Sleep Model for binary arousal detection  ##################
def build_usleep_model(input_shape=(134400, 2), alpha=1.67):
    def encoder_block(x, filters, kernel_size=9):
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        res = x
        x = layers.ZeroPadding1D((0, 1))(x) if x.shape[1] % 2 != 0 else x
        x = layers.MaxPooling1D(2)(x)
        return x, res

    def decoder_block(x, res, filters, kernel_size=9):
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        diff = res.shape[1] - x.shape[1]
        if diff > 0:
            res = layers.Cropping1D((diff // 2, diff - diff // 2))(res)
        elif diff < 0:
            x = layers.Cropping1D((-diff // 2, -diff - (-diff // 2)))(x)
        x = layers.Concatenate()([x, res])
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = inputs

    encoder_residuals = []
    filter_sizes = np.array([6, 9, 11, 15, 20, 28, 40, 55, 77, 108, 152, 214])

    for filters in filter_sizes:
        x, res = encoder_block(x, filters)
        encoder_residuals.append(res)
        x = layers.Conv1D(int(306 * np.sqrt(alpha)), 9, padding='same')(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)

    for res, filters in zip(reversed(encoder_residuals), reversed(filter_sizes)):
      x = decoder_block(x, res, filters)
       
      x = layers.Conv1D(6, 1, padding='same', activation='tanh')(x)
      x = layers.AveragePooling1D(pool_size=64)(x)
      x = layers.Conv1D(5, 1, padding='same', activation='elu')(x)
      x = layers.Conv1D(1, 1, padding='same', activation='sigmoid')(x)
      outputs = x

    model = keras.Model(inputs, outputs)
    return model

#####################        TFRecord generation function             #################
def save_all_segments_to_tfrecord(folder, tfrecord_path):
   
    h5_file = os.path.join(folder, f"{os.path.basename(folder)}.h5")
 
    with h5py.File(h5_file, 'r') as f:
     
        # Load EEG/EOG/ECG/EMG signals
        raw_signals = [f['signals'][channel][:] for channel in
                       [
                        'lat', 'rat']]

        #  Resample EEG/EOG/ECG/EMG signals
        print(f"Original Signal Shape (Before Resampling): {raw_signals[0].shape}")  
        fs_orig, fs_target = 200, 128
        signal = np.stack([resample_signal(sig, fs_orig, fs_target) for sig in raw_signals], axis=1)
        print(f"Resampled Signal Shape (After Resampling): {signal.shape}")        
        #  Preprocess EEG/EOG/ECG/EMG signals
        for i, channel_name in enumerate([
                                          'lat', 'rat']):
            signal[:, i, 0] = preprocess_signals(signal[:, i, 0], fs=fs_target, signal_name=channel_name)
    #  Segment Data into 30s Windows
        num_segments = signal.shape[0] // 3840
        valid_signals = [signal[i * 3840:(i + 1) * 3840, :] for i in range(num_segments)]
        if len(valid_signals) < 35:
           print(f"Skipping folder {folder}: Not enough valid segments")
           return        
       
        # Load and Resample Annotations
        annotations = f['annotations']['limb'][:]
        print(f"Original Annotations Shape: {annotations.shape}")        
        valid_annotations = resample_annotations_to_2Hz(annotations)
        print(f" Resampled Annotation Shape: {valid_annotations.shape}")
        print(f" Number of Arousals in Resampled Annotations: {np.sum(valid_annotations == 1)}")

               
    #  Writing TFRecord
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for i in range(0, len(valid_signals) - 35 + 1, 35):
                signals = np.concatenate(valid_signals[i:i + 35], axis=0)        
                annotations_segment = valid_annotations[i * 60: (i + 35) * 60]
                print(f"annotations_segment Shape: {annotations_segment.shape}")
                print(f" Number of Arousals in Annotations Segment: {np.sum(annotations_segment == 1)}")
                feature = {
                   'signals': tf.train.Feature(bytes_list=tf.train.BytesList(value=[signals.tobytes()])),
                   'annotations': tf.train.Feature(int64_list=tf.train.Int64List(value=annotations_segment)),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

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
    signals = tf.reshape(signals, (134400, 2))

    # Extract annotations
    annotations = tf.cast(parsed_example['annotations'], tf.int32)



    return signals, annotations


def create_dataset(tfrecord_files, batch_size=64, shuffle_buffer_size=100, prefetch_buffer_size=1):
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
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=2)  # Parse each record
    dataset = dataset.shuffle(shuffle_buffer_size)  # Shuffle dataset
    dataset = dataset.batch(batch_size)  # Batch the data
    dataset = dataset.prefetch(prefetch_buffer_size)  # Prefetch for performance
    return dataset




##### ***********************************EOFs*************************************** ########

####################################################################  CSV  ###############################################
root_dir = '/media/cdac/311f8483-7bbc-4353-a171-8eb9b61bf683/cdac Dropbox/BCH_h5_samples'

df_train = pd.read_csv('/media/cdac/lachesis/train_set.csv')
subID_train = df_train['subID'].to_numpy()
sess_train = df_train['Session'].to_numpy()
train_folders = []


for i in range(0,len(subID_train)):
    subID_sess_train = subID_train[i] + '_ses-' + str(sess_train[i])
    for fo in os.listdir(root_dir):
        if fo.startswith(subID_sess_train):
            train_folders.append(fo)

df_val = pd.read_csv('/media/cdac/lachesis/val_set.csv')
subID_val = df_val['subID'].to_numpy()
sess_val = df_val['Session'].to_numpy()
val_folders = []


for i in range(0,len(subID_val)):
    subID_sess_val = subID_val[i] + '_ses-' + str(sess_val[i])
    for fo in os.listdir(root_dir):
        if fo.startswith(subID_sess_val):
            val_folders.append(fo)

df_test = pd.read_csv('/media/cdac/lachesis/test_set.csv')
subID_test = df_test['subID'].to_numpy()
sess_test = df_test['Session'].to_numpy()
test_folders = []


for i in range(0,len(subID_test)):
    subID_sess_test = subID_test[i] + '_ses-' + str(sess_test[i])
    for fo in os.listdir(root_dir):
        if fo.startswith(subID_sess_test):
            test_folders.append(fo)

print(len(train_folders))
print(len(val_folders))
print(len(test_folders))


#######################      Dataset Creation       ##################################################################


# for folder in train_folders:
#     tfrecord_dir = os.path.join("/media/cdac/clotho/pedi caisr/limb movement", "train", folder)  # Train subdir
#     os.makedirs(tfrecord_dir, exist_ok=True)  # Create folder if missing
   
#     tfrecord_path = os.path.join(tfrecord_dir, f"{os.path.basename(folder)}.tfrecord")
#     if os.path.exists(tfrecord_path):
#         #print(f"Skipping {folder} - TFRecord already exists.")
#         continue
#     save_all_segments_to_tfrecord(os.path.join(root_dir, folder), tfrecord_path)  # Ensure full path

for folder in val_folders[0:100]:
    tfrecord_dir = os.path.join("/media/cdac/clotho/pedi caisr/limb movement", "val", folder)  # Val subdir
    os.makedirs(tfrecord_dir, exist_ok=True)  # Create folder if missing
   

    tfrecord_path = os.path.join(tfrecord_dir, f"{os.path.basename(folder)}.tfrecord")
    if os.path.exists(tfrecord_path):
        #print(f"Skipping {folder} - TFRecord already exists.")
        continue
    save_all_segments_to_tfrecord(os.path.join(root_dir, folder), tfrecord_path)  # Ensure full path

for folder in test_folders[0:100]:
    tfrecord_dir = os.path.join("/media/cdac/clotho/pedi caisr/limb movement", "test", folder)  # Test subdir
    os.makedirs(tfrecord_dir, exist_ok=True)  # Create folder if missing
   
    tfrecord_path = os.path.join(tfrecord_dir, f"{os.path.basename(folder)}.tfrecord")
    if os.path.exists(tfrecord_path):
        #print(f"Skipping {folder} - TFRecord already exists.")
        continue
    save_all_segments_to_tfrecord(os.path.join(root_dir, folder), tfrecord_path)  # Ensure full path


#train_folders = [f for f in os.listdir("/media/cdac/clotho/pedi caisr/limb movement/train") if os.path.isdir(os.path.join("/media/cdac/clotho/pedi caisr/limb movement/train", f))]
#val_folders = [f for f in os.listdir("/media/cdac/clotho/pedi caisr/limb movement/val") if os.path.isdir(os.path.join("/media/cdac/clotho/pedi caisr/limb movement/val", f))]
##test_folders = [f for f in os.listdir("/media/cdac/clotho/pedi caisr/limb movement/test") if os.path.isdir(os.path.join("/media/cdac/clotho/pedi caisr/limb movement/test", f))]

#train_tfrecords = [os.path.join("/media/cdac/clotho/pedi caisr/limb movement/train", f, f"{f}.tfrecord") for f in train_folders]
#train_dataset = create_dataset_arousal(train_tfrecords, batch_size=64)
#for signals, annotations in train_dataset.take(1):
    #print("Train data shape:", signals.shape, annotations.shape)
   
#val_tfrecords = [os.path.join("/media/cdac/clotho/pedi caisr/limb movement/val", f, f"{f}.tfrecord") for f in val_folders]
#val_dataset = create_dataset_arousal(val_tfrecords, batch_size=64)
#for signals, annotations in val_dataset.take(1):
    ##print("Validation data shape:", signals.shape, annotations.shape)

#test_tfrecords = [os.path.join("/media/cdac/clotho/pedi caisr/limb movement/test", f, f"{f}.tfrecord") for f in test_folders]
#test_dataset = create_dataset_arousal(test_tfrecords, batch_size=64)
#for signals, annotations in test_dataset.take(1):
#    print("Test data shape:", signals.shape, annotations.shape)

#tf.keras.backend.clear_session()
#train_folders = [os.path.join('train_orig', d) for d in os.listdir('train_orig')]
#train_tfrecords = [os.path.join(f, os.path.basename(f) + '.tfrecord') for f in train_folders]
#train_dataset = create_dataset(train_tfrecords, batch_size=64)  # smaller batch size

#print(len(train_tfrecords))

#val_folders = [os.path.join('val_orig', d) for d in os.listdir('val_orig')]
#val_tfrecords = [os.path.join(f, os.path.basename(f) + '.tfrecord') for f in val_folders]  
#val_dataset = create_dataset(val_tfrecords, batch_size=64)
#print(len(val_tfrecords))

#for signals, annotations in train_dataset.take(1):
#    print("Train data shape:", signals.shape, annotations.shape)
#for signals, annotations in val_dataset.take(1):
#    print("Validation data shape:", signals.shape, annotations.shape)
##tf.keras.backend.clear_session()

###################### Build and compile the model     ##################################################################

#model = build_usleep_model_ayt(input_shape=(134400, 10))
#print(model.summary())
#optimizer = keras.optimizers.Adam(learning_rate=0.0001)
##model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', f1_m])
#print("Model Built")
#print("Computing weights")
#class_weight_dict = compute_class_weights(train_dataset)
#print("Class weights computed")
#print("Model Complied")



#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=10, mode='max', restore_best_weights=True)
#f1_callback = F1ScoreCallback(validation_data=val_dataset, train_data=train_dataset)
#model.fit(train_dataset, validation_data=val_dataset, epochs=1000, callbacks=[f1_callback, early_stopping], verbose=1)


################################## Save Model ##################################

#model.save("trained_model_usleep_new.h5") 