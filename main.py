import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Available devices:")
for device in tf.config.list_physical_devices():
    print("-", device)

import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")