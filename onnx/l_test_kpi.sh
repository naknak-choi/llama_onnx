# 기본 실행
CUDA_VISIBLE_DEVICES=7 python l_test_kpi.py \
    --path ./onnx_llama3_2_3b_instruct_manual \
    --max_samples 100 \
    --gpu 7

CUDA_VISIBLE_DEVICES=7 python l_test_kpi.py \
    --path ./onnx_llama3_2_3b_instruct_lora_merged \
    --max_samples 100 \
    --gpu 7

CUDA_VISIBLE_DEVICES=7 python l_test_kpi.py \
    --path ./onnx_llama3_2_3b_instruct_custom \
    --max_samples 100 \
    --gpu 7
