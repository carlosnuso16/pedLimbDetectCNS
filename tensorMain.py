import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use only GPU 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
import glob # We'll use this to find your .tfrecord files
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
import warnings
import GPUtil

# --- 1. Constants (from your original script) ---
# These are needed for the parser to know the shape of your data
SEGMENT_COUNT = 35
SEGMENT_LEN_SEC = 30
SF_TARGET = 128
ANOT_TARGET_FREQ = 2
NUM_CHANNELS = 2 # Rat and Lat

SIGNAL_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * SF_TARGET # 134400
ANOT_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * ANOT_TARGET_FREQ # 2100

# --- 2. Data Loading & Parsing ---
# This section is for reading your *existing* .tfrecord files.

def parse_tfrecord(example_proto):
    """
    Parses a single TFRecord example.
    
    CRITICAL: This parser is from your *first* script. It correctly
    handles the signal shape you saved: (2, 134400). It will
    reshape and transpose it to (134400, 2) for the model.
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # 1. Decode signals (bytes to float32)
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    
    # 2. Reshape to (Channels, Samples)
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES)) # Reshape to (2, 134400)
    
    # 3. Transpose to (Time, Channels) for the 1D U-Net
    signals = tf.transpose(signals) # Transpose to (134400, 2)
    
    # 4. Extract and cast annotations
    annotations = tf.cast(parsed_example['annotations'], tf.int32)

    return signals, annotations

def create_dataset(tfrecord_files, batch_size=64, shuffle_buffer_size=100, prefetch_buffer_size=1, is_training=True):
    """
    Create a tf.data.Dataset pipeline from a list of TFRecord files.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    
    if is_training:
        # For training, shuffle the dataset
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.repeat()
        
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)  # Parse each record
    dataset = dataset.batch(batch_size)  # Batch the data
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for performance
    
    return dataset

# --- 3. Model Architecture ---
# This is the 'build_usleep_model_ayt' from your advisor's script.
# I've removed the buggy 'build_usleep_model' to avoid confusion.

def build_usleep_model_ayt(input_shape=(134400, 2), alpha=1.67):
    """
    Builds the 1D U-Net model.
    This is the corrected version ('ayt') from your advisor's script.
    """
    def encoder_block(x, filters, kernel_size=9):
        # NOTE: l2_lambda and dropout_rate are hardcoded to None
        # based on the provided script.
        l2_lambda = None
        dropout_rate = None
        
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        res = x
        # Pad if the time dimension is odd
        x = layers.ZeroPadding1D((0, 1))(x) if x.shape[1] % 2 != 0 else x
        x = layers.MaxPooling1D(2)(x)
        return x, res

    def decoder_block(x, res, filters, kernel_size=9):
        l2_lambda = None # Hardcoded
        
        kernel_regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
       
        # Crop or pad the residual connection (skip connection) to match shapes
        diff = res.shape[1] - x.shape[1]
        if diff > 0: # res is larger
            res = layers.Cropping1D((diff // 2, diff - diff // 2))(res)
        elif diff < 0: # x is larger
            x = layers.Cropping1D((-diff // 2, -diff - (-diff // 2)))(x)
       
        x = layers.Concatenate()([x, res])
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=kernel_regularizer)(x)
        x = layers.ELU()(x)
        x = layers.BatchNormalization()(x)
        return x

    # --- Model Body ---
    l2_lambda = None # Hardcoded
    inputs = keras.Input(shape=input_shape)
    x = inputs

    encoder_residuals = []
    # These are the 12 filter sizes for the 12 encoder blocks
    filter_sizes = np.array([6, 9, 11, 15, 20, 28, 40, 55, 77, 108, 152, 214])

    # Encoder Path
    for filters in filter_sizes:
        x, res = encoder_block(x, filters)
        encoder_residuals.append(res)

    # Bottleneck
    x = layers.Conv1D(int(306 * np.sqrt(alpha)), 9, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None)(x)
    x = layers.ELU()(x)
    x = layers.BatchNormalization()(x)

    # Decoder Path
    for res, filters in zip(reversed(encoder_residuals), reversed(filter_sizes)):
        x = decoder_block(x, res, filters)

    # --- Output Head ---
    # This maps the high-res (134400) output to the low-res (2100) target
    
    x = layers.Conv1D(6, 1, padding='same', activation='tanh')(x)
    
    # CRITICAL DOWNSAMPLING: 134400 / 64 = 2100
    x = layers.AveragePooling1D(pool_size=64)(x) 
    
    x = layers.Conv1D(5, 1, padding='same', activation='elu')(x)
    
    # Final prediction layer: (Batch, 2100, 1)
    outputs = layers.Conv1D(1, 1, padding='same', activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model

# --- 4. Loss Functions (for Imbalanced Data) ---

def compute_class_weights(train_dataset):
    """Calculate class weights based on training data distribution."""
    print("Calculating class weights... This may take a moment.")
    all_labels = []
   
    # Iterate over the dataset to collect all labels
    for _, annotations in train_dataset:
        all_labels.extend(annotations.numpy().flatten())

    # Compute weights
    class_labels = np.unique(all_labels)
    class_weights = compute_class_weight("balanced", classes=class_labels, y=all_labels)

    # Convert to dictionary format
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
   
    print("Computed Class Weights:", class_weight_dict)
    return class_weight_dict

def weighted_focal_loss(class_weight_dict, gamma=2.0):
    """
    A custom loss function that combines Focal Loss with
    pre-computed class weights.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Cast y_true to int to use as index
        y_true_int = tf.cast(y_true, 'int32')
       
        # Get the weights for each sample
        # [0.5, 56.5]
        weights_tensor = tf.constant([class_weight_dict[i] for i in sorted(class_weight_dict.keys())], dtype=tf.float32)
        # Gather weights: if y_true_int is 0, pick 0.5; if 1, pick 56.5
        weights = tf.gather(weights_tensor, y_true_int)

        # Compute focal loss components
        focal_loss_pt = -y_true * tf.math.log(y_pred) * (1 - y_pred) ** gamma
        focal_loss_npt = - (1 - y_true) * tf.math.log(1 - y_pred) * (y_pred ** gamma)
        
        focal_loss = focal_loss_pt + focal_loss_npt
        
        # Apply the class weights
        weighted_loss = weights * focal_loss
        
        return tf.reduce_mean(weighted_loss)

    return loss_fn

# --- 5. Metrics & Callbacks (for Imbalanced Data) ---

class F1ScoreCallback(Callback):
    """
    A custom callback to compute F1 score on training and validation
    data at the end of each epoch.
    
    This is much more reliable than 'accuracy' for imbalanced data.
    It also saves the model *only* when the validation F1 improves.
    """
    def __init__(self, validation_data, train_data=None):
        super().__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        self.best_f1 = 0.0  # Track best F1 score

    def compute_f1(self, dataset, dataset_name=""):
        """Helper function to compute F1 score for a given dataset."""
        print(f"\nCalculating F1 score for {dataset_name} data...")
        predictions = []
        true_labels = []
   
        # We must iterate over the whole dataset
        for signals, annotations in dataset:
            preds = self.model.predict(signals, verbose=0)
            preds_binary = (preds > 0.5).astype(int) # Convert probabilities to 0 or 1
            predictions.extend(preds_binary.flatten())
            true_labels.extend(annotations.numpy().flatten())

        # Use 'weighted' F1 to account for imbalance in the metric itself
        return f1_score(true_labels, predictions, average="binary", pos_label=1)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 1. Compute Validation F1
        val_f1 = self.compute_f1(self.validation_data, "validation")
        logs["val_f1_score"] = val_f1
        print(f"Epoch {epoch+1}: val_f1_score = {val_f1:.4f}")

        # 2. Optionally compute Training F1 (can be slow, set train_data=None to disable)
        if self.train_data:
            train_f1 = self.compute_f1(self.train_data, "training")
            logs["train_f1_score"] = train_f1
            print(f"Epoch {epoch+1}: train_f1_score = {train_f1:.4f}")

        # 3. Save the best model based on validation F1
        if val_f1 > self.best_f1:
            print(f"Validation F1 improved from {self.best_f1:.4f} to {val_f1:.4f}. Saving model.")
            self.best_f1 = val_f1
            self.model.save("best_model_by_f1.h5") # Save the best model
        else:
            print(f"Validation F1 did not improve from {self.best_f1:.4f}.")


# --- 6. Main Training Execution ---

if __name__ == "__main__":
    

    # directories containing your .tfrecord files
    tfFOLDERS = '/media/erikjan/SeagateC25_stora/pedLimbDetectCNS/tfrecords/'
    TRAIN_TFRECORD_DIR = tfFOLDERS + "train"
    VAL_TFRECORD_DIR = tfFOLDERS + "val"
    TEST_TFRECORD_DIR = tfFOLDERS + "test"

    BATCH_SIZE = 32 ## crashed at 8... trying 2
    
    print("--- 1. Finding TFRecord Files ---")
    # Use glob to find all .tfrecord files, including in subdirectories
    train_tfrecords = glob.glob(os.path.join(TRAIN_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    val_tfrecords = glob.glob(os.path.join(VAL_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    test_tfrecords = glob.glob(os.path.join(TEST_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    
    if not train_tfrecords or not val_tfrecords:
        raise FileNotFoundError("Could not find any .tfrecord files. Please check your TRAIN_TFRECORD_DIR and VAL_TFRECORD_DIR paths.")
        
    print(f"Found {len(train_tfrecords)} training TFRecord files.")
    print(f"Found {len(val_tfrecords)} validation TFRecord files.")
    print(f"Found {len(test_tfrecords)} test TFRecord files.")

    print("\n--- 2. Creating tf.data.Dataset Pipelines ---")
    # Note: We pass is_training=False to val/test to disable shuffling
    train_dataset = create_dataset(train_tfrecords, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = create_dataset(val_tfrecords, batch_size=BATCH_SIZE, is_training=False)
    test_dataset = create_dataset(test_tfrecords, batch_size=BATCH_SIZE, is_training=False)

    # (Optional) Check the output shape of one batch
    for signals, annotations in train_dataset.take(1):
        print(f"Batch shapes: Signals={signals.shape}, Annotations={annotations.shape}")
        # Should be: Signals=(64, 134400, 2), Annotations=(64, 2100)

    print("\n--- 3. Computing Class Weights ---")
    # This will iterate through the training dataset once
    # class_weight_dict = compute_class_weights(train_dataset)
    class_weight_dict = {0: 0.5018162877934619, 1: 138.14338498553548}
    print("Using Class Weights:", class_weight_dict)

    print("\n--- 4. Building and Compiling Model ---")
    tf.keras.backend.clear_session()
    model = build_usleep_model_ayt(input_shape=(SIGNAL_SAMPLES, NUM_CHANNELS))
    # model.summary() # Uncomment to see the full architecture
    
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    
    # Compile the model with our custom weighted focal loss
    model.compile(
        optimizer=optimizer,
        loss=weighted_focal_loss(class_weight_dict, gamma=2.0),
        metrics=['accuracy'], # We can monitor accuracy, but F1 is what matters
        run_eagerly=False
    )
    print("Model compiled successfully.")

    print("\n--- 5. Setting up Callbacks ---")
    # This callback will save the best model to 'best_model_by_f1.h5'
    f1_callback = F1ScoreCallback(
        validation_data=val_dataset,
        train_data=None # Set to `train_dataset` if you want F1 on training data (slower)
    )
    
    # This callback will stop training if the val_loss doesn't improve for 10 epochs
    # Note: Your advisor's code monitored 'val_f1_score', but that's not
    # automatically logged. We'll monitor 'val_loss' instead.
    # The F1 callback *already* saves the best model by F1 score.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=10, 
        mode='min', 
        restore_best_weights=True # Restore weights from epoch with best val_loss
    )
    #point of interest: lost
    print("\n--- 6. Starting Model Training ---")
    steps_per_epoch = 844 #len(train_tfrecords) // BATCH_SIZE
    validation_steps = 160 #len(val_tfrecords) // BATCH_SIZE

    print(f"--- 6. Starting Model Training ---")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Steps per Epoch: {steps_per_epoch} (train_files / batch_size)")
    print(f"   Validation Steps: {validation_steps} (val_files / batch_size)")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10000, # Set a high number; EarlyStopping will handle the rest
        callbacks=[f1_callback, early_stopping],
        verbose=1, # 'verbose=1' shows the progress bar
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    print("\n--- Training complete ---")
    print(f"Best validation F1 score achieved: {f1_callback.best_f1:.4f}")
    print("Best model saved to 'best_model_by_f1.h5'")

    # (Optional) Evaluate on the test set
    print("\n--- 7. Evaluating on Test Set ---")
    print("Loading best model from 'best_model_by_f1.h5'...")
    best_model = tf.keras.models.load_model(
        'best_model_by_f1.h5',
        custom_objects={'loss_fn': weighted_focal_loss(class_weight_dict)}
    )
    
    # Compute final F1 on test data
    test_f1 = f1_callback.compute_f1(test_dataset, "TEST")
    print(f"--- FINAL TEST F1 SCORE: {test_f1:.4f} ---")
