#'/mnt/SeagateC25_stora/pedLimbDetectCNS/tfrecords'
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tqdm import tqdm
import warnings

# --- 1. CONFIGURATION ---
# These must match your TFRecord creation
SF_TARGET = 128
ANOT_TARGET_FREQ = 2
NUM_CHANNELS = 2

SIGNAL_SAMPLES = 134400 # 17.5 mins of 128Hz samples
ANOT_SAMPLES = 2100     # 17.5 mins of 2Hz samples

# --- 2. TFRecord Parsing (CORRECTED PARSER) ---
def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example into signals and annotations.
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # 1. Decode signals as float32 (THIS IS THE FIX)
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    # 2. (REMOVED the tf.cast)
    
    # 3. Reshape and Transpose to (Time, Channels)
    #    This will now work, as 'signals' has 268,800 elements
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES))
    signals = tf.transpose(signals) # (134400, 2)
    
    # 4. Extract annotations
    annotations = tf.cast(parsed_example['annotations'], tf.int32)
    return signals, annotations

def create_dataset(tfrecord_files):
    """
    Creates a simple, non-batched, non-repeating dataset 
    to iterate through chunks.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# --- 3. Plotting Function ---
def plot_segment(signal_segment, annot_segment, segment_index):
    """
    Plots a single 30-second segment to visually check alignment.
    signal_segment shape: (3840, 2)
    annot_segment shape: (60,)
    """
    
    # Calculate lengths and time axis for plotting
    signal_len_samples = signal_segment.shape[0] # Should be 3840
    annot_len_samples = annot_segment.shape[0]   # Should be 60
    
    # Create a time axis in SECONDS for the 128Hz signal
    time_axis_signal = np.arange(signal_len_samples) / SF_TARGET
    
    # --- CRITICAL ALIGNMENT STEP ---
    # Upsample the 2Hz annotations to match the 128Hz signal time
    # 128Hz / 2Hz = 64. Each annotation point corresponds to 64 signal points.
    upsample_factor = SF_TARGET // ANOT_TARGET_FREQ
    annot_upsampled = np.repeat(annot_segment, upsample_factor)
    
    # Check if upsampling worked (should be 60 * 64 = 3840)
    if len(annot_upsampled) != signal_len_samples:
        # Handle potential off-by-one errors if data is not perfect
        annot_upsampled = np.pad(annot_upsampled, (0, signal_len_samples - len(annot_upsampled)), 'edge')

    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f'Alignment Check: Segment {segment_index} (30 seconds)', fontsize=16)

    # Plot 1: RAT Signal
    axs[0].plot(time_axis_signal, signal_segment[:, 0], label='RAT Signal (10Hz Filtered)', color='C0')
    axs[0].set_title('RAT Signal')
    axs[0].legend(loc='upper right')

    # Plot 2: LAT Signal
    axs[1].plot(time_axis_signal, signal_segment[:, 1], label='LAT Signal (10Hz Filtered)', color='C1')
    axs[1].set_title('LAT Signal')
    axs[1].legend(loc='upper right')

    # Plot 3: Annotations (as blocks)
    axs[2].fill_between(time_axis_signal, 0, annot_upsampled, 
                        label='Annotation (1=Movement)', 
                        color='red', alpha=0.5, step='post')
    axs[2].set_title('Annotation (Upsampled to 128Hz)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['No Movement', 'Movement'])
    axs[2].set_ylim(-0.1, 1.1)
    axs[2].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = f'alignment_check_{segment_index}.png'
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.close(fig)

# --- 4. Main Execution ---
def main():
    # --- IMPORTANT: Point this to your FAST SSD drive ---
    tfFOLDERS = '/mnt/SeagateC25_stora/pedLimbDetectCNS/tfrecords'
    
    TRAIN_TFRECORD_DIR = os.path.join(tfFOLDERS, "train")
    
    print("Finding training files...")
    # Use glob to find all .tfrecord files recursively
    train_tfrecords = glob.glob(os.path.join(TRAIN_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    
    if not train_tfrecords:
        raise FileNotFoundError(f"No files found at {TRAIN_TFRECORD_DIR}. Please check the path.")
        
    # We don't need to shuffle, just take the first N files
    dataset = create_dataset(train_tfrecords)

    plot_count = 0
    MAX_PLOTS = 10
    
    # Define the 30-second segment lengths
    signal_segment_len = SF_TARGET * 30  # 128 * 30 = 3840
    annot_segment_len = ANOT_TARGET_FREQ * 30 # 2 * 30 = 60
    
    print("Scanning for 30-second segments with movements...")

    # Loop over 17.5-minute chunks
    for signals_chunk, annots_chunk in tqdm(dataset):
        if plot_count >= MAX_PLOTS:
            break
            
        signals_np = signals_chunk.numpy()
        annots_np = annots_chunk.numpy()
        
        # Calculate how many 30s segments are in this chunk
        num_30s_segments = len(annots_np) // annot_segment_len # 2100 / 60 = 35

        # Loop over 30-second segments within the chunk
        for i in range(num_30s_segments):
            if plot_count >= MAX_PLOTS:
                break
                
            # Get the 30-second annotation segment
            a_start = i * annot_segment_len
            a_end = (i + 1) * annot_segment_len
            annot_segment = annots_np[a_start:a_end]
            
            # --- Check if this segment has any movement ---
            if np.any(annot_segment == 1):
                # Found one! Get the corresponding signal
                s_start = i * signal_segment_len
                s_end = (i + 1) * signal_segment_len
                signal_segment = signals_np[s_start:s_end]
                
                # Plot it
                plot_count += 1
                plot_segment(signal_segment, annot_segment, plot_count)
                
    print(f"\nDone. Found and plotted {plot_count} segments with movements.")

if __name__ == "__main__":
    # Suppress TensorFlow UserWarnings about data running out
    warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.keras.engine.data_adapter')
    main()