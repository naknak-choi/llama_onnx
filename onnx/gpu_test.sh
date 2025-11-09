# path : onnx_llama3_2_3b_instruct_lora_merged / onnx_llama3_2_3b_instruct_custom

CUDA_VISIBLE_DEVICES=7 python gpu_test.py \
  --path ./onnx_llama3_2_3b_instruct_custom \
  --gpu 0 \
  --prompt "The best thing about South Korea is" \
  --max_length 128 \
  --trials 5 \
  --warmups 1 \
  --arena_same


### custom
# === SUMMARY ===
# Δweights (avg±std, min~max): 6.016±0.106  5.963~6.229 GB
# Δrun     (avg±std, min~max): 0.097±0.003  0.096~0.104 GB


# merge
# === SUMMARY ===
# Δweights (avg±std, min~max): 6.782±0.106  6.729~6.994 GB
# Δrun     (avg±std, min~max): 0.097±0.003  0.096~0.104 GB
