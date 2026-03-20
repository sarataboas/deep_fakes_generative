import torch

def get_device():
    """
    Returns the available device in the following order: CUDA, MPS, CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device