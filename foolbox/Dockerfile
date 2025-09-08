# 1. 기본 이미지 설정
# 사용자님의 드라이버(CUDA 12.9 지원)는 CUDA 12.1 툴킷과 완벽하게 호환됩니다.
# PyTorch 2.1.0과 CUDA 12.1은 안정성이 검증된 조합입니다.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. 작업 디렉토리 설정
WORKDIR /workspace

# 3. 시스템 패키지 설치 (필요시)
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 4. Foolbox 및 필수 Python 라이브러리 설치
RUN pip install --no-cache-dir \
    foolbox \
    torchvision \
    matplotlib \
    jupyterlab

# 5. 컨테이너 실행 시 Jupyter Lab 실행 (선택 사항)
# 포트 8888을 외부에 개방하고, Jupyter Lab을 실행합니다.
EXPOSE 8888
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]