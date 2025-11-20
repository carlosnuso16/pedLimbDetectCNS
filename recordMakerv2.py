import os
import h5py
import numpy as np
import mne
import pandas as pd
import socket
from tqdm import tqdm
import tensorflow as tf
import warnings

# Suppress MNE logging
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# --- 1. Constants ---
SF_ORIG = 200 # Original sampling frequency in H5 files
SF_TARGET = 128 # Target sampling frequency for processing
ANOT_TARGET_FREQ = 2 # Target frequency for annotations

# Constants for TFRecord structure (35 segments * 30 seconds)
SEGMENT_COUNT = 35
SEGMENT_LEN_SEC = 30

# Calculate samples PER 30-SECOND SEGMENT (at original 200Hz)
CHUNK_LEN_SEC = 30 # We will process in 30s chunks
CHUNK_SAMPLES_200HZ = CHUNK_LEN_SEC * SF_ORIG # 30 * 200 = 6000

# Calculate FINAL samples PER 30-SECOND SEGMENT (after resampling)
CHUNK_SAMPLES_128HZ = CHUNK_LEN_SEC * SF_TARGET # 30 * 128 = 3840
CHUNK_SAMPLES_2HZ = CHUNK_LEN_SEC * ANOT_TARGET_FREQ # 30 * 2 = 60

# Calculate FINAL samples PER 17.5-MINUTE (35-segment) TFRECORD
SIGNAL_SAMPLES_FINAL = SEGMENT_COUNT * CHUNK_SAMPLES_128HZ # 35 * 3840 = 134400
ANOT_SAMPLES_FINAL = SEGMENT_COUNT * CHUNK_SAMPLES_2HZ   # 35 * 60 = 2100
NUM_CHANNELS = 2 # Rat and Lat


# --- 2. Helper Functions for Resampling (operate on 30s chunks) ---

def resample_signal_chunk(signal_chunk_200hz):
    """
    Resamples one 30-second, 200Hz signal chunk (2, 6000)
    down to a 128Hz signal chunk (2, 3840).
    """
    # Create a temporary MNE object for this 30s chunk
    info = mne.create_info(ch_names=['rat', 'lat'], sfreq=SF_ORIG, ch_types=['emg', 'emg'])
    raw_chunk = mne.io.RawArray(signal_chunk_200hz, info)
    
    # Resample to 128Hz
    raw_chunk.resample(SF_TARGET)
    
    data_128hz = raw_chunk.get_data()
    
    # Ensure correct shape (3840 samples)
    if data_128hz.shape[1] == CHUNK_SAMPLES_128HZ:
        return data_128hz.astype(np.float32)
    else:
        # Pad or truncate if MNE's resample gives a slightly different length
        target_shape = (NUM_CHANNELS, CHUNK_SAMPLES_128HZ)
        final_data = np.zeros(target_shape, dtype=np.float32)
        copy_len = min(data_128hz.shape[1], CHUNK_SAMPLES_128HZ)
        final_data[:, :copy_len] = data_128hz[:, :copy_len]
        return final_data

def resample_anot_chunk(anot_chunk_200hz):
    """
    Resamples one 30-second, 200Hz annotation chunk (6000,)
    down to a 2Hz annotation chunk (60,).
    """
    factor = int(SF_ORIG / ANOT_TARGET_FREQ) # 200 / 2 = 100
    
    # Ensure the chunk is the exact length (6000)
    if anot_chunk_200hz.shape[0] != CHUNK_SAMPLES_200HZ:
        # This shouldn't happen if our chunking logic is right
        return np.zeros(CHUNK_SAMPLES_2HZ, dtype=np.int64)

    # Reshape (6000,) -> (60, 100) and take the max over the 100 samples
    resampled = np.max(anot_chunk_200hz.reshape(-1, factor), axis=1)
    return resampled.astype(np.int64)


# --- 3. TFRecord Generation (Following the "Golden Rule") ---

def save_all_segments_to_tfrecord(folder, tfrecord_path):
    h5_file = os.path.join(folder, f"{os.path.basename(folder)}.h5")
    subject_id = os.path.basename(folder)

    try:
        with h5py.File(h5_file, 'r') as f:
            
            # 1. Load FULL 200Hz data
            raw_signals_list = [
                np.expand_dims(f['signals'][channel][:].squeeze(), axis=0)
                for channel in ['rat', 'lat']
            ]
            raw_signals_200hz = np.vstack(raw_signals_list) # Shape (2, N)
            annotations_200hz = f['annotations']['limb'][:] # Shape (N,)
            
            # 2. Alignment Sanity Check (at 200Hz)
            if raw_signals_200hz.shape[1] != annotations_200hz.shape[0]:
                print(f"SKIPPING {subject_id}: Raw signal and annotation lengths do not match!")
                print(f"  Signal: {raw_signals_200hz.shape[1]}, Anot: {annotations_200hz.shape[0]}")
                return

            # 3. Filter the FULL 200Hz signal
            info = mne.create_info(ch_names=['rat', 'lat'], sfreq=SF_ORIG, ch_types=['emg', 'emg'])
            raw = mne.io.RawArray(raw_signals_200hz, info)
            raw.filter(l_freq=10, h_freq=None, fir_design='firwin', picks='all')
            raw.notch_filter(60, picks='all')
            filtered_signals_200hz = raw.get_data()
            
            # 4. Find total number of 30s chunks
            num_30s_chunks = filtered_signals_200hz.shape[1] // CHUNK_SAMPLES_200HZ
            
            if num_30s_chunks < SEGMENT_COUNT: # 35
                print(f"SKIPPING {subject_id}: Not enough 30s chunks ({num_30s_chunks} < {SEGMENT_COUNT})")
                return

            # 5. Loop, Chunk, and Resample (The "Golden Rule")
            # We will store the resampled 30s chunks in these lists
            all_resampled_signals = []
            all_resampled_anots = []
            
            print(f"Processing {subject_id}: Found {num_30s_chunks} 30-second chunks...")
            for i in range(num_30s_chunks):
                # Get the 200Hz chunks
                s_start = i * CHUNK_SAMPLES_200HZ
                s_end = (i + 1) * CHUNK_SAMPLES_200HZ
                
                signal_chunk_200hz = filtered_signals_200hz[:, s_start:s_end]
                anot_chunk_200hz = annotations_200hz[s_start:s_end]
                
                # Resample them (now they are aligned)
                signal_chunk_128hz = resample_signal_chunk(signal_chunk_200hz)
                anot_chunk_2hz = resample_anot_chunk(anot_chunk_200hz)
                
                all_resampled_signals.append(signal_chunk_128hz)
                all_resampled_anots.append(anot_chunk_2hz)

            # 6. Write to TFRecord in 17.5-minute (35-segment) chunks
            if os.path.exists(tfrecord_path):
                print(f"Skipping writing: {tfrecord_path} already exists.")
                return

            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                # Loop over our 30s segments in steps of 35
                for i in range(0, len(all_resampled_signals) - SEGMENT_COUNT + 1, SEGMENT_COUNT):
                    
                    # Get the 35 segments from our lists
                    signal_list_35 = all_resampled_signals[i : i + SEGMENT_COUNT]
                    anot_list_35 = all_resampled_anots[i : i + SEGMENT_COUNT]
                    
                    # Stack signals: (35, 2, 3840) -> Transpose (2, 35, 3840) -> Reshape (2, 134400)
                    signals_final = np.stack(signal_list_35).transpose(1, 0, 2).reshape(NUM_CHANNELS, -1)
                    
                    # Concatenate annotations: (35, 60) -> (2100,)
                    anots_final = np.concatenate(anot_list_35)
                    
                    # --- Check for the bug we are fixing ---
                    if np.sum(anots_final) > 0 and signals_final.shape[1] != SIGNAL_SAMPLES_FINAL:
                         print(f"CRITICAL ERROR on {subject_id} chunk {i}: Signal shape mismatch!")
                         continue # Skip this bad chunk
                    
                    # Create the TFRecord feature
                    feature = {
                        'signals': tf.train.Feature(bytes_list=tf.train.BytesList(value=[signals_final.tobytes()])),
                        'annotations': tf.train.Feature(int64_list=tf.train.Int64List(value=anots_final)),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            
            print(f"Successfully wrote {subject_id} to {tfrecord_path}")

    except Exception as e:
        print(f"!!! FAILED processing {subject_id}: {e}")
        # If the file exists but failed, remove it so we can retry
        if os.path.exists(tfrecord_path):
            os.remove(tfrecord_path)

# --- 4. Main Execution Functions ---

def get_base_path():
    computer_name = socket.gethostname()
    if computer_name == "Flippy":
        return "c:/Users/carlo/"
    elif computer_name == "erikjan-desktop":
        return "/mnt/SeagateC25_stora/"
    else:
        print("Warning: Unknown computer, using default path.")
        return "/mnt/SeagateC25_stora/"

def load_h5_folders(csv_path, root_dir):
    df = pd.read_csv(csv_path)
    subIDs = df['subID'].to_numpy()
    sessions = df['Session'].to_numpy()
    folders = []
    
    h5_files_in_root = {fo for fo in os.listdir(root_dir) if fo.endswith(".h5")}

    for i in range(len(subIDs)):
        subID_sess = subIDs[i] + '_ses-' + str(sessions[i])
        # Find the folder that *starts with* this prefix
        found = False
        for fo in os.listdir(root_dir): # Check directories
            if os.path.isdir(os.path.join(root_dir, fo)) and fo.startswith(subID_sess):
                folders.append(fo)
                found = True
                break
        if not found:
             print(f"Warning: Could not find H5 *folder* for {subID_sess}")
                
    return folders

def process_data_set(folders, set_name, root_dir, base_tfrecord_output, limit=None):
    output_dir = os.path.join(base_tfrecord_output, set_name)
    os.makedirs(output_dir, exist_ok=True)

    if limit:
        folders = folders[:limit]

    print(f"\n--- Processing {set_name.capitalize()} Set ({len(folders)} files) ---")
    for folder in tqdm(folders, desc=f"Processing {set_name.capitalize()}"):
        
        subject_output_dir = os.path.join(output_dir, folder)
        os.makedirs(subject_output_dir, exist_ok=True)
        tfrecord_path = os.path.join(subject_output_dir, f"{folder}.tfrecord")

        # Call the save function
        save_all_segments_to_tfrecord(os.path.join(root_dir, folder), tfrecord_path)

def main_generate():
    base_path = get_base_path()
    root_dir = os.path.join(base_path, 'cdac Dropbox/BCH_h5_samples')
    scripts_dir = os.path.join(base_path, 'pedLimbDetectCNS')

    train_folders = load_h5_folders(os.path.join(scripts_dir, 'train_set.csv'), root_dir)
    val_folders = load_h5_folders(os.path.join(scripts_dir, 'val_set.csv'), root_dir)
    test_folders = load_h5_folders(os.path.join(scripts_dir, 'test_set.csv'), root_dir)

    print(f"Found {len(train_folders)} train, {len(val_folders)} val, {len(test_folders)} test folders.")

    BASE_TFRECORD_OUTPUT = os.path.join(scripts_dir, 'tfrecords')

    # Process all sets
    process_data_set(train_folders, "train", root_dir, BASE_TFRECORD_OUTPUT, limit=500)
    process_data_set(val_folders, "val", root_dir, BASE_TFRECORD_OUTPUT, limit=100)
    process_data_set(test_folders, "test", root_dir, BASE_TFRECORD_OUTPUT, limit=100)

    print("\nTFRecord generation complete.")

if __name__ == "__main__":
    main_generate()