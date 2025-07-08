import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        try:
            tensor_gpu = torch.tensor([1.0, 2.0, 3.0]).to(f'cuda:{i}')
            tensor_gpu.to('cpu')  # Move tensor back to CPU
            print(f"Successfully moved tensor to GPU {i} and back to CPU.")
        except Exception as e:
            print(f"Error moving tensor to GPU {i}: {e}")
else:
    print("CUDA is not available. Please check your installation.")
