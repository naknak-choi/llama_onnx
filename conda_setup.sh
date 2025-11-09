#!/bin/bash
# ==============================
# Conda 환경 생성 및 패키지 설치 스크립트
# ==============================

# 1. 새 conda 환경 생성 (Python 3.11)
conda create -y -n q9task python=3.11

# 2. 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate q9task

# 3. 필수 패키지 설치
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers==4.47.1
pip install peft==0.17.1
pip install nvidia-ml-py3
pip install lm_eval==0.4.8
pip install accelerate==1.6.0
pip install jsonlines==4.0.0
pip install ipykernel==6.29.5
pip install ipywidgets==8.1.5
pip install sentencepiece==0.2.0
pip install scipy==1.15.2
pip install matplotlib==3.10.1
pip install datasets==3.5.0
pip install tensorboard==2.19.0
pip install evaluate==0.4.3
pip install pyarrow==19.0.1
pip install onnx==1.18.0
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install optimum==1.26.1

