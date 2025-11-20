import tensorflow as tf
import os
import sys

print(f"--- TensorFlow and System Info ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")

print(f"\n--- CUDA and GPU ---")
print(f"Is TensorFlow built with CUDA?   {tf.test.is_built_with_cuda()}")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nFound {len(gpus)} Physical GPUs:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
            print(f"    Total Memory: {details.get('device_memory_mb', 'N/A')} MB")
    else:
        print(f"\n*** ERROR: tf.config.list_physical_devices('GPU') found 0 GPUs. ***")

except Exception as e:
    print(f"\nAn error occurred while checking for GPUs:")
    print(e)
    
print("\n--- CUDA_VISIBLE_DEVICES ---")
print(f"Environment variable CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print("\nDiagnosis complete.")