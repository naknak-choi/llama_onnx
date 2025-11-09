import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from transformers.cache_utils import DynamicCache
import torch.nn as nn
import os
from utils import load_compressed_model, count_parameters

# =================================================================================
# 1. 기본 설정 및 모델/토크나이저 로드
# =================================================================================

save_dir = Path("./onnx_llama3_2_3b_instruct_custom")
save_dir.mkdir(parents=True, exist_ok=True)
device = "cuda:0"

model_id = Path("your_custom_path/checkpoint-5000")
print(f"Loading model '{model_id}'...")
model = load_compressed_model(model_id, device)
for n, p in model.named_parameters():
    print(n)
print(count_parameters(model))
exit()
model.eval()
model.to("cuda")


# ONNX 변환에 필요한 모든 정보를 포함하는 config 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        # 입력 처리: 펼쳐진 텐서 -> DynamicCache 객체
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

        # to_legacy_cache()로 DynamicCache 객체를 다시 간단한 튜플로 변환
        present_legacy_tuple = outputs.past_key_values.to_legacy_cache()
        # 튜플을 펼쳐서 개별 텐서의 나열로 만듦
        flat_present = tuple(item for pair in present_legacy_tuple for item in pair)

        # 최종적으로 (logits, present_key_0, present_value_0, ...) 형태의 평탄한 튜플을 반환
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