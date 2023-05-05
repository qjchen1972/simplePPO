import os
import platform
import random
import re
import numpy as np
import torch 
import gym

def set_random_seed(seed, using_cuda=False):
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)
    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def get_device(device="auto"):
    # Cuda by default
    if device == "auto":
        device = "cuda"
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")
    return device

def safe_mean(arr):
    return np.nan if len(arr) == 0 else np.mean(arr)


def obs_as_tensor(obs, device):
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: torch.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)
    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)
    return scheduler
    
def get_system_info(print_info=True):
    """
    Retrieve system and python env info for the current system.

    :param print_info: Whether to print or not those infos
    :return: Dictionary summing up the version for each relevant package
        and a formatted string.
    """
    env_info = {
        # In OS, a regex is used to add a space between a "#" and a number to avoid
        # wrongly linking to another issue on GitHub. Example: turn "#42" to "# 42".
        "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "GPU Enabled": str(torch.cuda.is_available()),
        "Numpy": np.__version__,
        "Gym": gym.__version__,
    }
    env_info_str = ""
    for key, value in env_info.items():
        env_info_str += f"- {key}: {value}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str
    
    
def preprocess_obs(obs):
    obs = obs.permute(0,3,1,2)
    return obs.float() / 255.0
    
    
    
if __name__ == '__main__':
   get_system_info() 
   device = get_device()
   print(device, device.type)
   
   
   
