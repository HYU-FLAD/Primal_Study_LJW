import torch

# 1. 설치된 파이토치 버전 확인 (+cu118이 포함되어야 함)
print(torch.__version__)

# 2. CUDA (GPU) 사용 가능 여부 확인 (True가 나와야 함)
print(torch.cuda.is_available())