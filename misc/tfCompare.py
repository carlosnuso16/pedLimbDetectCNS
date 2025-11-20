import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import tensorflow as tf
from tqdm import tqdm
from tfRecordMake import get_base_path


# --- CONFIGURATION CONSTANTS (MUST MATCH TFRecord Generator) ---
FS_TARGET = 128         # Target sampling frequency for signals
ANOT_TARGET_FREQ = 2    # Target frequency for annotations
EPOCH_LEN_SEC = 30      # Length of the plot window (30 seconds)


SEGMENT_COUNT = 35      # Chunks of 35 segments per TFRecord (17.5 minutes)
SIGNAL_SAMPLES = 134400 # 35 segments * 3840 samples (Total samples per record)
ANOT_SAMPLES = 2100     # 35 segments * 60 samples (Total annotation samples per record)
NUM_CHANNELS = 2        # Rat and Lat


OUTPUT_ROOT = "tfrecord_comparison_plots"
PLOT_MAX_CHUNKS = 5      # Only process this many 17.5 minute TFRecord chunks


# --- TFRECORD PARSING FUNCTIONS (Copied from previous file) ---


def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example into signals and annotations.
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)


    # 1. Decode signals (bytes to float64, then cast to float32)
    # FIX: Decoding as float64 (8 bytes) ensures we read the correct number of elements,
    # then we cast to float32 (4 bytes) for efficient processing later.
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float64)
    signals = tf.cast(signals, tf.float32) # Cast to float32 for typical ML models
   
    # 2. Reshape signals to (Channels, Samples) and Transpose to (Time, Channels)
    # Saved as (2, 134400), Reshape assumes 268800 elements total
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES))
    signals = tf.transpose(signals) # (Time, Channels)
   
    # 3. Extract and cast annotations
    annotations = tf.cast(parsed_example['annotations'], tf.int32)


    return signals, annotations




def create_dataset(tfrecord_files):
    """Creates a tf.data.Dataset pipeline, optimized with cache and prefetch."""
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    # --- OPTIMIZATION ADDED HERE: CACHE and PREFETCH ---
    dataset = dataset.cache() # Cache data in memory after first read (great for plotting loops)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # Pre-load data while plotting occurs
    return dataset




# --- PLOTTING LOGIC (ADAPTED FOR NUMPY ARRAYS) ---


def plot_30sec_epoch(proc_epoch, anot_epoch, chunk_index, epoch_num, filename_base):
    # --- EDITED FUNCTION NAME & DOCSTRING ---
    """
    Plots a 30-second window of the processed signal and annotations.
   
    Args:
        proc_epoch (np.ndarray): Shape (3840, 2) - 30 seconds of signal
        anot_epoch (np.ndarray): Shape (60,) - 30 seconds of annotations
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # --- EDITED SUPERTITLE ---
    fig.suptitle(f"TFRecord Chunk {chunk_index} | 30s Epoch #{epoch_num} (128 Hz Processed Data)")


    # 1. Plot Processed RAT (Column 0 in proc_epoch)
    # proc_epoch shape is (Time, Channels) -> (N, 2)
    axs[0].plot(proc_epoch[:, 0], label='RAT', color='C0')
    axs[0].set_title('Processed RAT (128 Hz)')
    axs[0].legend(loc='upper right')
    axs[0].set_ylim(-500, 500)

    # 2. Plot Processed LAT (Column 1 in proc_epoch)
    axs[1].plot(proc_epoch[:, 1], label='LAT', color='C1')
    axs[1].set_title('Processed LAT (128 Hz)')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(-500, 500)


    # 3. Plot Annotations (Anot data is 2 Hz)
    # Repeat the 2Hz data to align with the 128Hz time axis for visualization
    upsample_factor = FS_TARGET // ANOT_TARGET_FREQ # 128 / 2 = 64
    anot_epoch_128hz = np.repeat(anot_epoch, upsample_factor)
    anot_masked = np.where(anot_epoch_128hz == 1, 0.5, np.nan) # Mask 0s for cleaner plot


    axs[2].plot(anot_masked, color='C2', linewidth=5)
    axs[2].set_title(f'Limb Movement Annotations ({ANOT_TARGET_FREQ} Hz)')
    # --- EDITED XLABEL ---
    axs[2].set_xlabel(f'Samples (Time axis: 30 sec * {FS_TARGET} Hz)')
    axs[2].set_ylim(-0.1, 1.1)
    axs[2].tick_params(axis='y', labelleft=False)


    plt.tight_layout()
   
    subject_folder = os.path.join(OUTPUT_ROOT, filename_base)
    os.makedirs(subject_folder, exist_ok=True)
    plt.savefig(os.path.join(subject_folder, f'chunk{chunk_index}_epoch_{epoch_num}.png'))
    plt.close()




def main_plot_tfrecords():
    """
    Main function to load TFRecords and generate plots.
    """
    # ------------------------------------------------------------------
    # 1. SETUP PATHS
    # ------------------------------------------------------------------

    BASE_TFRECORD_PATH = '/media/erikjan/SeagateC25_stora/cdac Dropbox/Carlos Nunez-Sosa/pedLimb0/tfrecords/train'
    # 
   
    if not os.path.exists(BASE_TFRECORD_PATH):
        print(f"ERROR: TFRecord path not found: {BASE_TFRECORD_PATH}")
        print("Please update BASE_TFRECORD_PATH to point to your saved 'val' folder.")
        return


    tfrecord_files = [
        os.path.join(BASE_TFRECORD_PATH, f)
        for f in os.listdir(BASE_TFRECORD_PATH)
        if f.endswith('.tfrecord')
    ]
   
    if not tfrecord_files:
        print(f"ERROR: No TFRecord files found in: {BASE_TFRECORD_PATH}")
        return


    # ------------------------------------------------------------------
    # 2. CREATE DATASET AND ITERATE
    # ------------------------------------------------------------------
    dataset = create_dataset(tfrecord_files)
   
    # Define slicing parameters for 30-second epochs
    # --- RECALCULATED SLICE LENGTHS ---
    signal_slice_len = int(EPOCH_LEN_SEC * FS_TARGET) # 3840 samples
    anot_slice_len = int(EPOCH_LEN_SEC * ANOT_TARGET_FREQ) # 60 samples
   
    # The whole chunk is 1050 seconds long (35 * 30), so we get 35 full 30s epochs.
    # --- RECALCULATED TOTAL EPOCHS ---
    total_epochs_per_chunk = SIGNAL_SAMPLES // signal_slice_len # 134400 / 3840 = 35


    print(f"Found {len(tfrecord_files)} TFRecords. Generating {total_epochs_per_chunk} plots per record.")
   
    chunk_count = 0
    for chunk_index, (signals_np, annotations_np) in enumerate(tqdm(dataset.as_numpy_iterator(), total=len(tfrecord_files), desc="Plotting TFRecord Chunks")):
       
        if chunk_count >= PLOT_MAX_CHUNKS:
            break
           
        # signals_np shape: (134400, 2) - Time x Channels
        # annotations_np shape: (2100,) - Annotation Time


        file_base = os.path.basename(tfrecord_files[chunk_index]).replace('.tfrecord', '')
       
        # Iterate over 30-second epochs within the 17.5-minute chunk
        for epoch_num in range(total_epochs_per_chunk):
           
            s_start = epoch_num * signal_slice_len
            s_stop = (epoch_num + 1) * signal_slice_len
           
            a_start = epoch_num * anot_slice_len
            a_stop = (epoch_num + 1) * anot_slice_len


            proc_epoch = signals_np[s_start:s_stop, :]
            anot_epoch = annotations_np[a_start:a_stop]
           
            # --- EDITED FUNCTION CALL ---
            plot_30sec_epoch(proc_epoch, anot_epoch, chunk_index, epoch_num, file_base)


        chunk_count += 1
   
    print(f"\nSuccessfully generated plots in the '{OUTPUT_ROOT}' directory.")


if __name__ == "__main__":
    main_plot_tfrecords()
