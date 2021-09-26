#This function will check which processor exists.
import torch
def pro_check(gpu_arg):
    if not gpu_arg: return torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu": 
        print("CUDA is not on available, using CPU on this device")
    else:
        print("CUDA is available on this device")
    return device