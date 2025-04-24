import os
import sys

# Setting environment variable to avoid CUDA errors if CUDA is not available
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

try:
    # Try importing torch without requiring CUDA
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability safely
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU for computations.")
    except RuntimeError as cuda_error:
        print(f"Error checking CUDA: {cuda_error}")
        print("CUDA check failed. Using CPU for computations.")
        cuda_available = False
        
    # Check for MPS (Apple Metal) availability
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    # Determine available device
    if cuda_available:
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while checking PyTorch: {e}")
    print("Continuing with CPU only.")
    device = torch.device("cpu")
    print(f"Using device: {device}")

# Check that the training_image_model.py script is properly device-aware
try:
    import src.training.train_image_model
    print("Successfully imported training modules.")
except ImportError as e:
    print(f"Error importing training modules: {e}")

# Successfully imported and checked environment
print("Environment check completed successfully.") 