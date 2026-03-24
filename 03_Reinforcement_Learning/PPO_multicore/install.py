import sys
import subprocess
import os

# 현재 파일(install.py)이 있는 폴더로 작업 위치를 고정합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print(f"📍 현재 작업 경로: {os.getcwd()}")
print("🚀 GPU용 환경 설정을 시작합니다...")

# 1. 기존 CPU용 패키지 제거
print("🧹 기존 CPU용 패키지 제거 중...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
except:
    pass

# 2. requirements.txt 설치 (경로를 파일 이름만 적으면 됩니다)
print("📦 기본 라이브러리 설치 중...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# 3. GPU용 PyTorch(CUDA 12.1) 설치
print("🔥 GPU용 PyTorch 설치 중 (약 2~3GB, 잠시만 기다려주세요)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "torch", "torchvision", "torchaudio", 
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

print("\n✅ 모든 설치가 완료되었습니다! 이제 GPU를 사용할 수 있습니다. 😎")