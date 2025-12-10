import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import groupby
import matplotlib.patches as mpatches

# --- 1. CONFIGURATION ---
tfFOLDERS = '/mnt/SeagateC25_stora/pedLimbDetectCNS/tfrecords/'
TEST_TFRECORD_DIR = os.path.join(tfFOLDERS, "test")
MODEL_PATH = "best_model_custom_loop.keras"

# Set this to None to enable "Scanner Mode" (find patients WITH labels)
# Set to a string (e.g. "1001_ses-1") to force a specific patient.
TARGET_PATIENT_ID = None 

# Constants
SF_TARGET = 128
ANOT_TARGET_FREQ = 2
NUM_CHANNELS = 2
SIGNAL_SAMPLES = 134400 
ANOT_SAMPLES = 2100 

# --- 2. The Plotting Function (Provided by You) ---
def multiple_lines_plot_masks_channel(subject, trace, PLM_expert, y, fs, normalize_signal=False, minutes_per_row=10, row_height=30):
    """
    Plots signals broken into rows (e.g. 10 mins per row).
    Expects trace, PLM_expert, and y to have the SAME length.
    """
    # Calculate dimensions
    minutes_data = trace.shape[0]/fs/60
    nrow = int(minutes_data/minutes_per_row) + 1
    
    # Dynamic figure size based on duration
    figsize = (18, 3 * nrow) 
    
    if normalize_signal:
        # Simple robust normalization
        trace = (trace - np.median(trace, axis=0)) / (np.quantile(trace, 0.75, axis=0) - np.quantile(trace, 0.25, axis=0))

    # Split indices into rows
    row_ids = np.array_split(np.arange(len(trace)), nrow)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    label_color = [None,'red'] # Prediction color
    label_color_expert = [None,'green'] # Expert color
        
    for ri in range(nrow):
        if len(row_ids[ri]) == 0: continue

        # --- 1. Plot Signals (RAT and LAT) ---
        # RAT (Channel 0)
        ax.plot(trace[row_ids[ri], 0] + 1.5*2*ri * row_height, c='k', lw=0.5, alpha=0.7)
        # LAT (Channel 1) - Offset by +10
        ax.plot(trace[row_ids[ri], 1] + 1.5*2*ri * row_height + 10, c='k', lw=0.5, alpha=0.7)
        
        # --- 2. Plot AI Predictions (Red) ---
        loc = 0
        # groupby clusters consecutive labels (e.g., 0,0,1,1,1,0 -> groups of 0s, 1s, 0s)
        for i, j in groupby(y[row_ids[ri]]):
            len_j = len(list(j))
            if not np.isnan(i) and int(i) == 1: # If label is 1 (Movement)
                # Plot a red bar below the signal
                ax.plot([loc, loc + len_j], [1.5* 2*ri * row_height - 4*row_height//4] * 2, c='r', lw=5) 
            loc += len_j
        
        # --- 3. Plot Human Expert Labels (Green) ---
        loc = 0
        for i, j in groupby(PLM_expert[row_ids[ri]]):
            len_j = len(list(j))
            if not np.isnan(i) and int(i) == 1:
                # Plot a green bar slightly above the red one
                ax.plot([loc, loc + len_j], [1.5* 2* ri * row_height - 3*row_height//4] * 2, c='g', lw=5) 
            loc += len_j
        
        # Add Y-axis labels for this row
        ticklocs = [ri * row_height * 3, ri * row_height * 3 + 10] # Approximation for visual placement
        
    # Legend and Cleanup
    pop_b = mpatches.Patch(color='r', label='AI Prediction')
    pop_a = mpatches.Patch(color='g', label='Human Expert')
    ax.legend(handles=[pop_a, pop_b], bbox_to_anchor=(1, 1))
    ax.set_title(f"Subject: {subject}")
    ax.axis('off')
    plt.tight_layout()

    return fig

# --- 3. Data Loading Helpers ---

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

def load_full_patient_data(tfrecord_path):
    """
    Reads a TFRecord file and concatenates ALL chunks into one continuous night.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    full_signals = []
    full_annotations = []
    
    # print("Loading and stitching data chunks...")
    for sig, anot in dataset: # Removed tqdm here to reduce clutter in loop
        full_signals.append(sig.numpy())
        full_annotations.append(anot.numpy())
        
    # Concatenate into one big array
    # Signals: (N_chunks, 134400, 2) -> (Total_Time, 2)
    stitched_signals = np.concatenate(full_signals, axis=0)
    # Annotations: (N_chunks, 2100) -> (Total_Time_2Hz,)
    stitched_annotations = np.concatenate(full_annotations, axis=0)
    
    return stitched_signals, stitched_annotations

# --- 4. Main Execution ---

def main():
    # 1. Find Files
    all_files = glob.glob(os.path.join(TEST_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
    
    if not all_files:
        print("No test files found!")
        return

    # Filter if target ID is set
    if TARGET_PATIENT_ID:
        files_to_process = [f for f in all_files if TARGET_PATIENT_ID in f]
        if not files_to_process:
            print(f"Could not find patient with ID: {TARGET_PATIENT_ID}")
            return
    else:
        files_to_process = all_files
        print(f"Scanning {len(files_to_process)} test files for subjects WITH expert labels...")

    # Load Model ONCE
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    plots_generated = 0
    MAX_PLOTS = 5 # Stop after finding 5 good examples

    # Loop through files
    for file_path in tqdm(files_to_process, desc="Scanning Subjects"):
        if plots_generated >= MAX_PLOTS:
            print(f"Generated {MAX_PLOTS} plots. Stopping.")
            break

        subject_name = os.path.basename(file_path).replace(".tfrecord", "")
        
        # 2. Load Data
        full_signals, full_annotations_2hz = load_full_patient_data(file_path)
        
        # --- CRITICAL CHECK: Does this patient actually have labels? ---
        total_movements = np.sum(full_annotations_2hz)
        if total_movements == 0:
            # print(f"Skipping {subject_name}: No expert labels found (clean night).")
            continue # Skip to the next patient
        
        print(f"\n>>> Found Subject with Labels: {subject_name} ({total_movements} movements)")

        # 3. Predict (Only run if we found labels!)
        batch_size = 32
        n_chunks = full_signals.shape[0] // SIGNAL_SAMPLES
        model_input = full_signals.reshape(n_chunks, SIGNAL_SAMPLES, NUM_CHANNELS)
        
        preds_prob = model.predict(model_input, batch_size=batch_size, verbose=0)
        full_predictions_2hz = (preds_prob > 0.5).astype(int).flatten()
        
        # 4. ALIGNMENT (Upsampling)
        upsample_factor = SF_TARGET // ANOT_TARGET_FREQ # 64
        
        full_annotations_128hz = np.repeat(full_annotations_2hz, upsample_factor)
        full_predictions_128hz = np.repeat(full_predictions_2hz, upsample_factor)
        
        # Safety Check
        min_len = min(len(full_signals), len(full_annotations_128hz), len(full_predictions_128hz))
        full_signals = full_signals[:min_len]
        full_annotations_128hz = full_annotations_128hz[:min_len]
        full_predictions_128hz = full_predictions_128hz[:min_len]

        # 5. Plotting
        print(f"Generating plot for {subject_name}...")
        fig = multiple_lines_plot_masks_channel(
            subject=subject_name,
            trace=full_signals,
            PLM_expert=full_annotations_128hz,
            y=full_predictions_128hz,
            fs=SF_TARGET,
            normalize_signal=True,
            minutes_per_row=10 
        )
        
        output_filename = f"full_night_{subject_name}.png"
        fig.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
        plt.close(fig) # Close memory
        
        plots_generated += 1

    if plots_generated == 0:
        print("\nWARNING: Scanned all files but found NO expert labels in any of them!")

if __name__ == "__main__":
    main()