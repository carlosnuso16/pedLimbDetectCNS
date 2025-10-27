import tensorflow as tf
import numpy as np
import os
from typing import Union
import h5py
import mne


# --- Configuration Constants ---
# These must match the constants used when the TFRecord was created.
SEGMENT_COUNT = 35
SEGMENT_LEN_SEC = 30
SF_TARGET = 128
ANOT_TARGET_FREQ = 2
NUM_CHANNELS = 2  # 'rat' and 'lat'
SF_ORIG = 200

# Calculated total samples per record
SIGNAL_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * SF_TARGET      # 134400
ANOT_SAMPLES = SEGMENT_COUNT * SEGMENT_LEN_SEC * ANOT_TARGET_FREQ # 2100

def parse_tfrecord(example_proto: tf.Tensor) -> tf.Tensor:
    """
    Parses a single serialized TFRecord example to extract the signals tensor.
    """
    feature_description = {
        'signals': tf.io.FixedLenFeature([], tf.string),
        'annotations': tf.io.FixedLenFeature([ANOT_SAMPLES], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    signals = tf.io.decode_raw(parsed_example['signals'], tf.float32)
    signals = tf.reshape(signals, (NUM_CHANNELS, SIGNAL_SAMPLES))
    signals = tf.transpose(signals)
    return signals


def get_rat_signal_from_tfrecord(tfrecord_path: str) -> Union[list[np.ndarray], None]:
    """
    Extracts all 'rat' signal arrays from a given TFRecord file.
    """
    if not os.path.exists(tfrecord_path):
        print(f"❌ Error: File not found at '{tfrecord_path}'")
        return None
    try:
        print(f"Processing file: {tfrecord_path}")
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        parsed_dataset = dataset.map(parse_tfrecord)
        rat_signals_list = []
        for signals_tensor in parsed_dataset:
            rat_signal_tensor = signals_tensor[:, 0]
            rat_signals_list.append(rat_signal_tensor.numpy())
        print(f"  - Extracted {len(rat_signals_list)} 'rat' signal array(s) from the file.")
        return rat_signals_list
    except Exception as e:
        print(f"An error occurred while processing the TFRecord file: {e}")
        return None
def get_h5_resampled_len(h5_file_path: str) -> Union[int, None]:
    """
    Loads the 'rat' signal from an H5 file, resamples it to 128 Hz,
    and returns its total length in samples.

    Args:
        h5_file_path: The full path to the .h5 file.

    Returns:
        The total number of samples in the resampled signal, or None on error.
    """
    print(f"\nAnalyzing original H5 file: {h5_file_path}")
    if not os.path.exists(h5_file_path):
        print(f"❌ H5 file not found at '{h5_file_path}'")
        return None
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # 1. Load the raw signal at its original 200 Hz
            raw_rat_signal = f['signals']['rat'][:].squeeze()
            print(f"  - Loaded original signal with {len(raw_rat_signal):,} samples at {SF_ORIG} Hz.")

            # 2. Create MNE object to handle resampling
            info = mne.create_info(ch_names=['rat'], sfreq=SF_ORIG, ch_types=['emg'])
            mne_raw = mne.io.RawArray([raw_rat_signal], info)

            # 3. Resample the data to the target frequency
            mne_raw.resample(SF_TARGET)

            # 4. Get the length of the resampled data
            resampled_len = mne_raw.n_times
            print(f"  - Resampled signal has {resampled_len:,} samples at {SF_TARGET} Hz.")
            return resampled_len
            
    except Exception as e:
        print(f"An error occurred while processing the H5 file: {e}")
        return None

# --- Main execution block to demonstrate the function ---
if __name__ == "__main__":
    # !!! IMPORTANT !!!
    # You must provide BOTH file paths for the comparison to work.
    base = "/media/erikjan/SeagateC25_stora/cdac Dropbox/Carlos Nunez-Sosa/pedLimb0/tfrecords/test/"
    example_tfrecord_file = base + "sub-I0003175001794_ses-1_task-PSG_eeg/sub-I0003175001794_ses-1_task-PSG_eeg.tfrecord"
    corresponding_h5_file = "/media/erikjan/SeagateC25_stora/cdac Dropbox/BCH_h5_samples/sub-I0003175001794_ses-1_task-PSG_eeg/sub-I0003175001794_ses-1_task-PSG_eeg.h5"

    all_rat_signals = get_rat_signal_from_tfrecord(example_tfrecord_file)

    if all_rat_signals:
        # --- Total Data Calculation (from TFRecord) ---
        samples_per_example = all_rat_signals[0].shape[0]
        num_examples = len(all_rat_signals)
        total_tfrecord_samples = samples_per_example * num_examples

        # --- H5 vs. TFRecord Length Comparison ---
        total_h5_samples = get_h5_resampled_len(corresponding_h5_file)

        if total_h5_samples is not None:
            discarded_samples = total_h5_samples - total_tfrecord_samples
            discarded_seconds = discarded_samples / SF_TARGET
            
            print("\n--- 📊 Final Comparison ---")
            print(f"Total potential length from H5 file: {total_h5_samples:,} samples")
            print(f"Total packaged length in TFRecord:   {total_tfrecord_samples:,} samples")
            print(f"Samples discarded during packaging:    {discarded_samples:,} samples ({discarded_seconds:.2f} seconds)")

    else:
        print("\n❌ Failure. Please check the TFRecord file path.")