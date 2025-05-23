import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    # Use an explicit device index
    props = torch.cuda.get_device_properties(0)
    print("CUDA Device Properties:", props)
else:
    print("No CUDA detected, proceeding with CPU.")
