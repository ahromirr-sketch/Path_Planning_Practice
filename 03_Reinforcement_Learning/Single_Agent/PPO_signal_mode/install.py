import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "03_Reinforcement_Learning/Single_Agent/PPO_signal_mode/requirements.txt"])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "torch", "torchvision", "torchaudio", 
    "--index-url", "https://download.pytorch.org/whl/cu121"
])