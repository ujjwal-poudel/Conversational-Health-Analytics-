import torch

# Checking if MPS is available
if torch.backends.mps.is_available():
    print("PyTorch has found the MPS backend. Your GPU is available!")
    
    # 2. Set the device to MPS
    device = torch.device("mps")
    
    # 3. Create a tensor and move it to the GPU
    print("\nCreating a tensor on the CPU...")
    x_cpu = torch.rand(3, 5)
    print(x_cpu)

    print("\nMoving the tensor to the GPU (MPS device)...")
    x_gpu = x_cpu.to(device)
    print(x_gpu)
    
else:
    print("MPS backend not available. PyTorch will use the CPU.")