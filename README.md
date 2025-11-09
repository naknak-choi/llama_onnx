# code 실행 가이드

`./` 디렉터리는 LLaMA 3.2 3B 기반 모델을 압축·미세조정하고 ONNX 런타임으로 배포·평가하기 위한 스크립트를 모아둔 실행 패키지입니다. 이 문서는 환경 준비부터 학습, ONNX 변환, 성능 검증까지의 전체 흐름을 다룹니다.

## 디렉터리 한눈에 보기

```
./
├── conda_setup.sh                # conda 환경 생성 및 핵심 패키지 설치 스크립트
├── train/
│   ├── training_lightweights.py  # SQuAD 기반 선택적 MLP 압축 + 지식증류 학습 파이프라인
│   └── training_lightweights_lora.py  # LoRA 기반 경량 학습 파이프라인
└── onnx/
    ├── original_torch2onnx.py          # LoRA 병합 모델 ONNX 변환
    ├── original_torch2onnx_custom.py   # 압축 모델 전용 ONNX 변환 (custom loader)
    ├── l_test_kpi.py                   # ONNX 모델 정확도/지연 시간 측정 (SQuAD)
    ├── gpu_test.py                     # ONNX 모델 GPU 메모리 사용량 측정
    ├── utils.py                        # 압축 모델 로더 및 유틸 함수
    ├── l_test_kpi.sh, gpu_test.sh      # 실행 예시 스크립트
```

## 필수 요구 사항

- **GPU**: NVIDIA GPU (권장 24GB 이상, CUDA 12.x 및 Driver 호환)
- **OS**: Linux 기반 환경 (테스트 기준 Ubuntu 20.04+)
- **Python**: 3.11 (스크립트에서 conda 환경 생성 시 자동 설치)
- **Hugging Face 계정**: `meta-llama/Llama-3.2-3B-Instruct` 모델 다운로드를 위해 토큰 필요
- **디스크 공간**: 모델 가중치 및 ONNX 산출물을 위한 수십 GB 여유 공간

> ⚠️ 사내/폐쇄망 환경에서는 Hugging Face 토큰과 외부 저장소 접근 권한을 확인하세요.

## 1. 환경 구성

### 1-1. conda 환경 자동 생성

최초 1회, `./` 루트에서 다음 명령을 실행합니다.

```bash
bash conda_setup.sh
```

스크립트는 `llama_onnx`라는 이름의 conda 환경을 만들고 PyTorch 2.5.1, CUDA 12.4, transformers 4.47.1, optimum 1.26.1 등 필요한 패키지를 설치합니다. 설치 중 오류가 발생하면 `conda info --envs`로 환경이 생성되었는지 확인한 뒤, 부족한 패키지를 수동으로 설치하세요.

### 1-2. 환경 활성화 & Hugging Face 로그인

```bash
conda activate llama_onnx
huggingface-cli login  # 토큰 입력
```

로그인 후 `huggingface-cli whoami`로 인증 상태를 확인합니다.

## 2. 모델 및 데이터 준비

- 학습 스크립트는 Hugging Face Datasets의 **SQuAD**를 자동 다운로드합니다. 프록시 환경에서는 `HF_ENDPOINT` 또는 `http_proxy` 환경 변수를 설정하세요.
- 기본 모델은 `meta-llama/Llama-3.2-3B-Instruct`입니다. 사전 다운로드 후 `cache_dir/`를 지정하고 싶다면 `HF_HOME` 또는 `TRANSFORMERS_CACHE`를 설정하세요.
- LoRA 가중치나 압축 모델 가중치 경로는 스크립트 상단 변수로 지정되어 있으므로, 본인 경로에 맞게 수정해야 합니다.

## 3. 학습 파이프라인

### 3-1. 선택적 MLP 압축 + KD (`train/training_lightweights.py`)

이 스크립트는 MLP 레이어를 SVD 기반으로 압축한 뒤, SQuAD 데이터셋으로 지식증류(KD) 학습을 수행합니다.

핵심 설정 (스크립트 상단):

- `model_name`: 교사/학생 모델 모두에 사용할 HF 허브 경로
- `rank_ratio`: SVD 보존 비율 (0.5 → 50% 랭크 유지)
- `compress_gate / compress_up / compress_down`: 각 프로젝션 압축 여부
- `alpha`, `temperature`: KD 손실 비중 및 소프트닝 정도
- `output_dir`: 학습 결과 저장 위치 (기본 `./compressed_llama_squad_sft_kd_selective`)

실행 예:

```bash
conda activate llama_onnx
python train/training_lightweights.py
```

실행 후 주요 산출물:

- `compressed_llama_initial_sft_selective/`: 압축 직후 학생 모델 및 `compression_config.json`
- `compressed_llama_squad_sft_kd_selective/final_model/`: KD 학습 완료 모델
- `tensorboard` 로그: `compressed_llama_squad_sft_kd_selective/logs`

> ✅ 학습 전에 GPU 메모리 여유(≥24GB)를 확보하고, 필요 시 `per_device_train_batch_size` 또는 `gradient_accumulation_steps`를 조정하세요.

### 3-2. LoRA 경량 학습 (`train/training_lightweights_lora.py`)

LoRA 어댑터만 학습하고 싶은 경우 사용합니다. 스크립트 상단에서 다음을 확인하세요.

- `base_model_id`: 베이스 모델 ID
- `lora_r`, `lora_alpha`, `target_modules`: LoRA 설정
- `output_dir`: 어댑터 가중치를 저장할 경로 (`lora_checkpoint-5000/` 등)

실행:

```bash
conda activate llama_onnx
python train/training_lightweights_lora.py
```

출력 디렉터리는 `--output_dir` 값에 생성됩니다. 이후 ONNX 변환 시 `lora_adapter_id`로 사용합니다.

## 4. ONNX 변환 및 배포 준비

### 4-1. LoRA 병합 모델 ONNX 변환 (`onnx/original_torch2onnx.py`)

1. 스크립트 상단 변수를 본인 경로에 맞게 수정합니다.
	 - `base_model_id = "meta-llama/Llama-3.2-3B-Instruct"`
	 - `lora_adapter_id = "<LoRA_가중치_경로>"`
	 - `save_dir`: ONNX 출력 디렉터리 (예: `./onnx_llama3_2_3b_instruct_lora_merged`)
2. 명령 실행:

```bash
conda activate llama_onnx
cd ./onnx
python onnx/original_torch2onnx.py
```

ONNX 파일(`decoder_model.onnx`)과 토크나이저/설정 파일이 `save_dir`에 저장됩니다.

### 4-2. 압축 모델 ONNX 변환 (`onnx/original_torch2onnx_custom.py`)

압축된 학생 모델을 로드하기 위해 `utils.load_compressed_model`을 사용합니다.

1. `model_id = Path("<압축_모델_경로>")`를 실제 경로로 변경 (`compressed_llama_squad_sft_kd_selective/final_model` 등)
2. 스크립트 중간에 있는 `exit()`는 파라미터 확인용입니다. 변환을 진행하려면 해당 줄을 주석 처리하거나 삭제하세요.
3. 필요 시 `device = "cuda:0"`를 조정합니다.
4. 실행:

```bash
conda activate llama_onnx
cd ./onnx
python original_torch2onnx_custom.py
```

완료 후 `save_dir`에 ONNX 파일과 토크나이저 설정이 생성됩니다.

### 4-3. 변환 시 참고 사항

- ONNX export는 GPU 메모리를 많이 사용합니다. 다른 프로세스를 종료한 뒤 실행하세요.
- `utils.py`의 `count_parameters` 함수를 이용해 압축 전후 파라미터 수를 비교할 수 있습니다.
- 변환 중 `pynvml` 관련 오류가 나면 `pip install nvidia-ml-py3`가 설치되었는지 확인하세요.

## 5. ONNX 모델 평가 및 모니터링

### 5-1. 정확도·지연 시간 측정 (`onnx/l_test_kpi.py`)

이 스크립트는 SQuAD 검증 세트를 기반으로 F1, Exact Match, 토큰 당 지연 시간(ms/token)을 계산합니다.

주요 인자:

- `--path`: Optimum-ONNX 모델 디렉터리 (토크나이저 파일 포함)
- `--max_samples`: 평가할 샘플 수 (기본 100)
- `--batch_size`: 배치 크기 (기본 1)
- `--output_dir`: 결과 JSON 저장 위치 (기본 `./evaluation_results`)

실행 예:

```bash
conda activate llama_onnx
cd ./onnx
CUDA_VISIBLE_DEVICES=0 python l_test_kpi.py \
	--path ./onnx_llama3_2_3b_instruct_custom \
	--max_samples 100 \
	--batch_size 1 \
	--gpu 0 \
	--output_dir ./evaluation_results
```

출력:

- `evaluation_results/<모델명>_evaluation_summary_000.json`
- `evaluation_results/<모델명>_sample_predictions.json`

### 5-2. GPU 메모리 사용량 측정 (`onnx/gpu_test.py`)

ONNX 모델의 prefill 단계에서 GPU 메모리 증가량을 반복 측정합니다.

- `--arena_same`: CUDAExecutionProvider에서 `kSameAsRequested` 전략 사용
- `--gpu_mem_limit_gb`: ORT GPU 할당 상한 설정

실행 예 (쉘 스크립트 참고):

```bash
conda activate llama_onnx
cd ./onnx
CUDA_VISIBLE_DEVICES=0 python gpu_test.py \
	--path ./onnx_llama3_2_3b_instruct_custom \
	--gpu 0 \
	--prompt "The best thing about South Korea is" \
	--max_length 128 \
	--trials 5 \
	--warmups 1 \
	--arena_same
```