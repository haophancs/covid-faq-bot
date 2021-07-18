import argparse
import os
import subprocess
import sys
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-type',
                        default='cpu',
                        choices=['gpu', 'cpu'],
                        required=False,
                        type=str)
    args = parser.parse_args()
    if args.device_type == 'cpu':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_cpu.txt"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_gpu.txt"])

    from dotenv import dotenv_values, find_dotenv, set_key
    import torch

    if args.device_type == 'gpu':
        assert torch.cuda.is_available()

    copyfile('./.env_org', './.env')
    dotenv_file = find_dotenv()
    all_env_values = dotenv_values(dotenv_file)
    for key, value in all_env_values.items():
        if "PATH" in key:
            value = os.path.join(os.path.abspath(os.getcwd()), value)
            set_key(dotenv_file, key, value)
    set_key(dotenv_file, "DEFAULT_DEVICE", "cuda:0" if args.device_type == 'gpu' else "cpu")
