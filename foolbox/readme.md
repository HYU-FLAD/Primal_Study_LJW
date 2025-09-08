# Foolbox 실행을 위한 Docker 환경

**이 프로젝트는 머신러닝 모델의 취약점 분석 라이브러리인 ****Foolbox**를 GPU를 활용하여 격리된 Docker 환경에서 실행하기 위한 설정을 제공합니다. Jupyter Lab이 포함되어 있어 웹 브라우저를 통해 편리하게 코드를 작성하고 실행할 수 있습니다.

## 사전 준비 사항

**이 환경을 사용하기 전에 아래의 프로그램들이 반드시 설치 및 설정되어 있어야 합니다.**

* **OS** **: Windows 10/11 (WSL 2 지원 버전)**
* **GPU** **: NVIDIA GPU**
* **드라이버** **: 최신 NVIDIA 드라이버**
* **Docker** **: Docker Desktop for Windows (WSL 2 백엔드 엔진 활성화 필수)**

## 🚀 빠른 시작 (Quick Start)

**터미널(PowerShell 또는 CMD)을 열고, **`Dockerfile`이 위치한 이 프로젝트의 루트 디렉토리에서 아래의 명령어들을 순서대로 실행하세요.

### 1. 도커 이미지 빌드 (Build Docker Image)

**먼저, **`Dockerfile`을 기반으로 `my-foolbox:latest`라는 이름의 도커 이미지를 생성합니다. 이 과정은 최초 한 번만 수행하면 됩니다.

```
docker build -t my-foolbox:latest .
```

> **참고:** 빌드 과정은 네트워크 환경에 따라 수 분이 소요될 수 있습니다.

### 2. 도커 컨테이너 실행 (Run Docker Container)

**빌드된 이미지를 사용하여 GPU가 할당된 도커 컨테이너를 실행합니다.**

```
docker run -it --rm --gpus all -p 8888:8888 -v "%CD%:/workspace" my-foolbox:latest
```

**명령어 옵션 설명:**

* `-it`: 컨테이너와 상호작용 가능한 터미널을 사용합니다.
* `--rm`: 컨테이너를 중지하면 자동으로 삭제하여 시스템을 깨끗하게 유지합니다.
* `--gpus all`: 호스트 PC의 **모든 NVIDIA GPU를 컨테이너에 할당**합니다. (가장 중요!)
* `-p 8888:8888`: 내 PC의 8888번 포트를 컨테이너의 8888번 포트(Jupyter Lab)와 연결합니다.
* `-v ${PWD}:/workspace`: 현재 로컬 폴더를 컨테이너의 `/workspace` 폴더와 동기화합니다. 로컬에서 파일을 수정하면 즉시 컨테이너에 반영됩니다.

### 3. Jupyter Lab 접속

**컨테이너가 성공적으로 실행되면, 사용하고 계신 웹 브라우저를 열고 아래 주소로 접속하세요.**

```
http://localhost:8888
```

**이제 Jupyter Lab 환경에서 자유롭게 Python 코드를 작성하고 Foolbox를 활용한 분석을 시작할 수 있습니다. 컨테이너를 중지하려면 터미널에서 **`Ctrl + C`를 누르세요.


Federated Learning에서의 Backdoor Attack을 적용하기에는 비적합하다고 판단, 폐기
