import tensorflow as tf
import numpy as np
import os
import glob
from tqdm import tqdm # A nice progress bar
import matplotlib.pyplot as plt

# --- Import functions from your training script ---
# (Make sure train_model_stable.py is in the same folder)
from tensorMain import (
    parse_tfrecord, 
    create_dataset,
    SIGNAL_SAMPLES,
    SF_TARGET # 128 Hz
)

# --- 1. Settings ---
TRAIN_TFRECORD_DIR = "/media/erikjan/SeagateC25_stora/pedLimbDetectCNS/tfrecords/train"
BATCH_SIZE = 8 # Keep this reasonable
SAMPLES_PER_WINDOW = int(SF_TARGET / 2) # 128 / 2Hz = 64 samples per window

def root_mean_square(signal):
    """Helper function to calculate RMS (a measure of power)"""
    return np.sqrt(np.mean(np.square(signal)))

# --- 2. Load Train Dataset ---
print("Finding train TFRecord files...")
train_tfrecords = glob.glob(os.path.join(TRAIN_TFRECORD_DIR, "**", "*.tfrecord"), recursive=True)
if not train_tfrecords:
    raise FileNotFoundError("No train files found!")

print(f"Loading {len(train_tfrecords)} files...")
# We set batch_size=1 to process one 17.5-min chunk at a time
# This makes the windowing logic easier
train_dataset = create_dataset(train_tfrecords, batch_size=1, is_training=False)

# --- 3. Loop and Calculate Stats ---
power_when_0 = []
power_when_1 = []

print("Analyzing dataset signal power... (This will take a while)")

# Use tqdm for a progress bar
for (signals_batch, annotations_batch) in tqdm(train_dataset):
    # We are using batch_size=1, so squeeze the batch dimension
    signals = np.squeeze(signals_batch.numpy(), axis=0) # Shape (134400, 2)
    annotations = np.squeeze(annotations_batch.numpy(), axis=0) # Shape (2100,)

    # We will just analyze the first channel (RAT) for simplicity
    # You could average them or do both later
    signal_ch1 = signals[:, 0]
    
    # Loop through the 2100 annotation labels
    for i in range(len(annotations)):
        label = annotations[i]
        
        # Find the corresponding signal window
        start_sample = i * SAMPLES_PER_WINDOW
        end_sample = (i + 1) * SAMPLES_PER_WINDOW
        
        # Get the 64 samples from the signal
        signal_window = signal_ch1[start_sample:end_sample]
        
        # Calculate power (RMS)
        power = root_mean_square(signal_window)
        
        # Add the power value to the correct list
        if label == 0:
            power_when_0.append(power)
        else:
            power_when_1.append(power)

print("\n--- Dataset Analysis Complete ---")

# --- 4. Print Results ---
print(f"--- Stats for 'No Movement' (Label 0) ---")
print(f"  Count:  {len(power_when_0)}")
print(f"  Mean Power:   {np.mean(power_when_0):.4f}")
print(f"  Median Power: {np.median(power_when_0):.4f}")
print(f"  Std Dev:      {np.std(power_when_0):.4f}")
print(f"  Max Power:    {np.max(power_when_0):.4f}")


print(f"\n--- Stats for 'Movement' (Label 1) ---")
print(f"  Count:  {len(power_when_1)}")
print(f"  Mean Power:   {np.mean(power_when_1):.4f}")
print(f"  Median Power: {np.median(power_when_1):.4f}")
print(f"  Std Dev:      {np.std(power_when_1):.4f}")
print(f"  Max Power:    {np.max(power_when_1):.4f}")

# --- 5. Plot Histogram ---
print("\nSaving histogram to 'power_histogram.png'...")
plt.figure(figsize=(12, 7))
# Use log=True to see the distributions more clearly, especially if they are skewed
plt.hist(power_when_0, bins=100, alpha=0.7, label='Label 0 (No Movement)', density=True, log=True)
plt.hist(power_when_1, bins=100, alpha=0.7, label='Label 1 (Movement)', density=True, log=True)
plt.legend()
plt.title('Distribution of Signal Power (RMS) for Each Label', fontsize=16)
plt.xlabel('Signal Power (RMS)')
plt.ylabel('Density (Log Scale)')
plt.savefig('power_histogram.png')
print("Done.")
