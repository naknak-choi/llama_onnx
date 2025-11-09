import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from transformers.cache_utils import DynamicCache
import torch.nn as nn
import os
from peft import PeftModel, LoraConfig

# =================================================================================
# 1. 기본 설정 및 모델/토크나이저 로드
# =================================================================================

# 1-1. 베이스 모델과 LoRA 어댑터 경로를 명확히 구분합니다.
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # LoRA 튜닝의 기반이 된 원본 모델
lora_adapter_id = "your_path/lora_checkpoint-5000" # LoRA 가중치만 있는 경로
save_dir = Path("./onnx_llama3_2_3b_instruct_lora_merged")
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading base model: '{base_model_id}'...")
# 1-2. 먼저 원본 '베이스 모델'을 로드합니다.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    # device_map="auto" # 필요 시 사용
)

# 토크나이저는 베이스 모델의 것을 사용합니다.
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print(f"Applying LoRA adapter: '{lora_adapter_id}'...")
# 1-3. 로드된 베이스 모델 위에 LoRA 어댑터를 적용합니다.
model = PeftModel.from_pretrained(base_model, lora_adapter_id)
config = model.config

model.eval()
model.to("cuda")

print("Merging LoRA weights into the base model...")
merged_model = model.merge_and_unload()
print("Merge complete. The model is now a standard Hugging Face model.")

merged_model.eval()
merged_model.to("cuda")

config = model.config
num_layers = config.num_hidden_layers
num_key_value_heads = config.num_key_value_heads
num_heads = config.num_attention_heads
dtype = torch.float16
device = model.device

hidden_size = config.hidden_size
head_dim = hidden_size // config.num_attention_heads

class LlamaOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, *past_key_values_flat):
        past_key_values_tuple = tuple(zip(past_key_values_flat[::2], past_key_values_flat[1::2]))
        past_key_values_cache = DynamicCache.from_legacy_cache(past_key_values=past_key_values_tuple)

        # 진짜 모델 호출
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values_cache,
            use_cache=True,
        )

        present_legacy_tuple = outputs.past_key_values.to_legacy_cache()
        flat_present = tuple(item for pair in present_legacy_tuple for item in pair)

        return (outputs.logits,) + flat_present

onnx_export_model = LlamaOnnxWrapper(model).eval()

# =================================================================================
# 2. ONNX 파일 1: KV 캐시가 없는 첫 번째 추론용 모델 (decoder_model.onnx)
# =================================================================================
print("\n--- Exporting Model for First Inference (without KV cache) ---")
batch_size = 1
sequence_length = 1
past_sequence_length = 16
total_sequence_length = past_sequence_length + sequence_length

print("ONNX 그래프 추적을 위한 더미 입력 생성")
dummy_input_ids = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)
dummy_attention_mask = torch.ones(batch_size, total_sequence_length, dtype=torch.long, device=device)
dummy_position_ids = torch.arange(
    past_sequence_length, total_sequence_length, dtype=torch.long, device=device
).unsqueeze(0)

dummy_past_key_values = tuple(
    (
        torch.randn(batch_size, num_key_value_heads, past_sequence_length, head_dim, dtype=torch.float16, device=device),
        torch.randn(batch_size, num_key_value_heads, past_sequence_length, head_dim, dtype=torch.float16, device=device),
    )
    for _ in range(num_layers)
)

print("입출력 이름과 동적 축을 정의")
input_names = ["input_ids", "attention_mask", "position_ids"]
output_names = ["logits"]
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
    "position_ids": {0: "batch_size", 1: "sequence_length"},
    "logits": {0: "batch_size", 1: "sequence_length"},
}

for i in range(num_layers):
    input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_sequence_length"}
    dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_sequence_length"}
    
    output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "total_sequence_length"}
    dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "total_sequence_length"}


# --- 5. torch.onnx.export 실행 (이전과 동일) ---
args_tuple = (dummy_input_ids, dummy_attention_mask, dummy_position_ids) + tuple(
    item for pair in dummy_past_key_values for item in pair
)

print("\nONNX 변환 시작")

# ONNX 변환 실행
torch.onnx.export(
    onnx_export_model,
    args=args_tuple,
    f=str(save_dir / "decoder_model.onnx"),
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=17,
    do_constant_folding=True
)
print("✅ Successfully exported decoder_model.onnx")


# =================================================================================
# 4. 추론에 필요한 토크나이저 및 설정 파일 저장
# =================================================================================
print("\n--- Saving tokenizer and configuration files ---")
tokenizer.save_pretrained(save_dir)
config.save_pretrained(save_dir)

print(f"\n[OK] All files saved successfully to: {save_dir}")
print("Directory contents:")
for file_path in save_dir.glob("*"):
    print(f"- {file_path.name}")