import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available: {cuda_avail}")

if cuda_avail:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {mem:.2f} GB")
    x = torch.randn(100, 100, device="cuda")
    print(f"Tensor computation on GPU test: {x.sum().item():.2f}")
    print("SUCCESS: GPU and CUDA are working properly!")
else:
    print("CUDA is NOT available.")
