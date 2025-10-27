import os
import h5py
import numpy as np
from scipy import stats as st
import mne
import pandas as pd
import socket
from tqdm import tqdm
import tensorflow as tf
mne.set_log_level('WARNING')


SF_ORIG = 200 # Original sampling frequency in H5 files
SF_TARGET = 128 # Target sampling frequency for processing
ANOT_TARGET_FREQ = 2 # Target frequency for annotations


# Constants for TFRecord structure (35 segments * 30 seconds)
SEGMENT_COUNT = 35
SEGMENT_LEN_SEC = 30
SIGNAL_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * SF_TARGET # 134400
ANOT_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * ANOT_TARGET_FREQ # 2100
NUM_CHANNELS = 2 # Rat and Lat


#####################        Preprocessing functions             #################


def preprocess_signals(signals, sfreq, resample_freq):
    ### MODIFY HERe FOR CHANGING prePROCESSING STEPS
    signals.filter(l_freq=10, h_freq=None, fir_design='firwin', picks='all', verbose=False)
    signals.notch_filter(60, picks='all', verbose=False)
    signals.resample(resample_freq, verbose=False)
    return signals


def resample_anot(anot, orig_sfreq=200, target_freq=2):
    factor = int(orig_sfreq / target_freq)
    n_samples = len(anot)
    trimmed_len = n_samples // factor * factor
    resampled = np.max(anot[:trimmed_len].reshape(-1, factor), axis=1)


    return resampled


#####################        TFRecord generation function             #################
def save_all_segments_to_tfrecord(folder, tfrecord_path):
    h5_file = os.path.join(folder, f"{os.path.basename(folder)}.h5")
    subject_id = os.path.basename(folder) # Added for file naming/logging


    with h5py.File(h5_file, 'r') as f:
       
        # Load EMG signals (RAT and LAT)
        # FIX: Ensure each signal is 2D (1, N) before stacking using np.expand_dims
        raw_signals_list = [
            np.expand_dims(f['signals'][channel][:].squeeze(), axis=0)
            for channel in ['rat', 'lat']
        ]
        raw_signals = np.vstack(raw_signals_list) # Resulting shape MUST be (2, N)
       
        # 1. Create MNE Info (Uses SF_ORIG=200 Hz)
        info = mne.create_info(
            ch_names=['rat', 'lat'],
            sfreq=SF_ORIG,
            ch_types=['emg', 'emg']
        )
        rawSignals =  mne.io.RawArray(raw_signals, info, verbose=False)


        # 2. Resample & preprocess EMG signals
        procSignals = preprocess_signals(rawSignals, sfreq=SF_ORIG, resample_freq=SF_TARGET)
       
        # Extract the data array from the MNE object (shape: [2, N'])
        proc_data = procSignals.get_data()
        signalShape = proc_data.shape[1] # Time dimension length


        print(f"Processed Signals Time Length: {signalShape}")


        # 3. Segment the signals (OPTIMIZED FOR EFFICIENCY)
        segmentLen = (SF_TARGET * SEGMENT_LEN_SEC) # 3840 samples per 30s segment
        num_segments = signalShape // segmentLen
        total_valid_samples = num_segments * segmentLen
       
        # Trim the full processed data to remove any partial segment at the end
        trimmed_proc_data = proc_data[:, :total_valid_samples]


        # Reshape into a 3D NumPy view (Segments, Channels, Samples) - ZERO COPY
        segmented_signals = trimmed_proc_data.reshape(num_segments, NUM_CHANNELS, segmentLen)
       
        if segmented_signals.shape[0] < SEGMENT_COUNT:
            print(f"Skipping folder {folder}: Not enough valid segments ({segmented_signals.shape[0]} < {SEGMENT_COUNT})")
            return
       
        # 4. Load and Resample Annotations
        annotations = f['annotations']['limb'][:]
        valid_annotations = resample_anot(annotations)


        # FIX: The print statements below are now correctly inside the 'with' block
        print(f" Resampled Annotation Shape: {valid_annotations.shape}")
        print(f" Number of Limb Movements in Resampled Annotations: {np.sum(valid_annotations == 1)}")


        # --- TFRECORD WRITING (Single File per Subject, Appending Chunks) ---
       
        # This is the single path all chunks will write to (appends all 35-segment chunks)
        # The calling code (below the function definition) is responsible for creating subject_output_dir
        # and setting tfrecord_path correctly.
       
        # Skip if the final combined file already exists
        if os.path.exists(tfrecord_path):
            print(f"Skipping writing: {subject_id}.tfrecord already exists.")
            return


        # Open the writer outside the segment loop so all chunks are appended to ONE file
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            # Loop over the 3D array in chunks of 35
            for i in range(0, segmented_signals.shape[0] - SEGMENT_COUNT + 1, SEGMENT_COUNT):
               
                # 1. Prepare data chunk
                segments_chunk = segmented_signals[i:i + SEGMENT_COUNT, :, :]
               
                # Reshape and flatten: (Channels, Total Samples)
                signals = segments_chunk.transpose(1, 0, 2).reshape(NUM_CHANNELS, -1).astype(np.float32)
               
                # Slice the corresponding annotation segment (35 * 60 = 2100 samples)
                anot_samples_per_segment = SEGMENT_LEN_SEC * ANOT_TARGET_FREQ # 60
                start_anot = i * anot_samples_per_segment
                end_anot = (i + SEGMENT_COUNT) * anot_samples_per_segment
                annotations_segment = valid_annotations[start_anot:end_anot]


                # 2. Write TFRecord Example
                feature = {
                    'signals': tf.train.Feature(bytes_list=tf.train.BytesList(value=[signals.tobytes()])),
                    'annotations': tf.train.Feature(int64_list=tf.train.Int64List(value=annotations_segment.astype(np.int64))),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
               
                print(f"Appended chunk {i} to {i + SEGMENT_COUNT - 1} to {subject_id}.tfrecord")




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
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)


    # 1. Decode signals (bytes to float32)
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    # 2. Reshape signals to (Channels, Samples) and Transpose to (Time, Channels)
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES)) # Reshape to (2, 134400)
    signals = tf.transpose(signals) # Transpose to (134400, 2) for (Time, Channels)
    # 3. Extract and cast annotations
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


def get_base_path():
    computer_name = socket.gethostname()
    if computer_name == "Flippy":
        return "c:/Users/carlo/"
    elif computer_name == "erikjan-desktop":
        return "/media/erikjan/SeagateC25_stora/"
    else:
        return "default/path/"


# --- Data Loading (Finding H5 Folders) ---
def load_h5_folders(csv_path, root_dir):
    df = pd.read_csv(csv_path)
    subIDs = df['subID'].to_numpy()
    sessions = df['Session'].to_numpy()
    folders = []


    for i in range(len(subIDs)):
        subID_sess = subIDs[i] + '_ses-' + str(sessions[i])
        for fo in os.listdir(root_dir):
            if fo.startswith(subID_sess):
                folders.append(fo)
    return folders



# --- PROCESSING LOOP TEMPLATE ---
def process_data_set(folders, set_name, limit=None):
    output_dir = os.path.join(BASE_TFRECORD_OUTPUT, set_name)
    os.makedirs(output_dir, exist_ok=True)


    if limit:
        folders = folders[0:limit]


    for folder in tqdm(folders, desc=f"Processing {set_name.capitalize()} Files"):
        # The 'save_all_segments_to_tfrecord' function expects the full subject H5 folder path
        # and the base output directory for the set (e.g., /tfrecords/train).
       
        # 1. Define the subject-specific output directory and file path
        subject_output_dir = os.path.join(output_dir, folder)
        os.makedirs(subject_output_dir, exist_ok=True) # Create folder for the subject
        tfrecord_path = os.path.join(subject_output_dir, f"{folder}.tfrecord")


        # 2. Call the save function
        save_all_segments_to_tfrecord(os.path.join(root_dir, folder), tfrecord_path)

def main_generate():
    root_dir = get_base_path()+'cdac Dropbox/BCH_h5_samples'
    scripts_dir = get_base_path()+'cdac Dropbox/Carlos Nunez-Sosa/pedLimb'

    train_folders = load_h5_folders(scripts_dir+'/train_set.csv', root_dir)
    val_folders = load_h5_folders(scripts_dir+'/val_set.csv', root_dir)
    test_folders = load_h5_folders(scripts_dir+'/test_set.csv', root_dir)


    print(len(train_folders))
    print(len(val_folders))
    print(len(test_folders))

    BASE_TFRECORD_OUTPUT = scripts_dir + '/tfrecords'

    process_data_set(train_folders, "train", limit=500)
    process_data_set(val_folders, "val", limit=100)
    process_data_set(test_folders, "test", limit=100)


    print("TFRecord generation complete.")



if __name__ == "__main__":
    main_generate()





