import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from safetensors.torch import load_file
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import re
import string
import logging
from tqdm import tqdm
import argparse
from collections import Counter
import os
import glob
from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path
import onnx

# === NEW: timing & NVML ===
import time
try:
    import pynvml  # pip install nvidia-ml-py3
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from onnx import TensorProto

def _pick_onnx_file(model_dir: str | Path) -> str:
    model_dir = Path(model_dir)
    for name in ("decoder_with_past_model.onnx", "decoder_model.onnx", "model.onnx"):
        p = model_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"ONNX file not found under {model_dir}")

def _sum_initializer_bytes(onnx_path: str) -> int:
    m = onnx.load(onnx_path, load_external_data=True)
    total = 0
    for t in m.graph.initializer:
        # 요소 개수
        n = 1
        for d in t.dims:
            n *= int(d)
        # dtype별 byte-per-element
        dt = t.data_type
        if   dt in (TensorProto.FLOAT16, TensorProto.BFLOAT16):  bpe = 2
        elif dt in (TensorProto.FLOAT, TensorProto.FLOAT32):     bpe = 4
        elif dt == TensorProto.DOUBLE:                           bpe = 8
        elif dt == TensorProto.INT8:                             bpe = 1
        elif dt in (TensorProto.UINT16, TensorProto.INT16):      bpe = 2
        elif dt in (TensorProto.UINT32, TensorProto.INT32):      bpe = 4
        elif dt in (TensorProto.UINT64, TensorProto.INT64):      bpe = 8
        else:                                                    bpe = 4  # 보수적 기본값
        total += n * bpe
    return total

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def extract_answer_from_generation(generated_text, prompt_text):
    answer = generated_text.split('Answer:')[-1].strip()
    answer = answer.replace('</s>', '').replace('<|endoftext|>', '').strip()
    answer = answer.rstrip('.')
    answer = answer.split('\n')[0].strip()
    return answer


# === NEW: GPU tracker (NVML) ===
class GPUTracker:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.enabled = _NVML_OK
        self.handle = None
        self.peak_used = 0  # bytes
        self.samples = []

    def __enter__(self):
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception as e:
                logger.warning(f"NVML init failed: {e}; GPU tracking disabled")
                self.enabled = False
        return self

    def sample(self):
        if not self.enabled: 
            return None
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used = int(mem.used)  # bytes
            self.samples.append(used)
            if used > self.peak_used:
                self.peak_used = used
            return used
        except Exception:
            return None

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def summary(self):
        if not self.enabled or not self.samples:
            return None
        avg = int(sum(self.samples) / len(self.samples))
        return {
            "peak_used_bytes": self.peak_used,
            "avg_used_bytes": avg,
            "peak_used_GB": self.peak_used / (1024**3),
            "avg_used_GB": avg / (1024**3),
            "num_samples": len(self.samples),
        }


def evaluate_model(
    model,
    tokenizer,
    dataset,
    model_name="Model",
    max_samples=1000,
    batch_size=4,
    max_new_tokens=5,
):
    """Evaluate a model on SQuAD validation set + throughput & GPU usage."""

    logger.info(f"\nEvaluating {model_name}...")

    # ORTModelForCausalLM에는 .device가 없을 수 있으니 안전 처리
    try:
        device = model.device  # torch 모델이면 존재
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f1_scores = []
    exact_scores = []
    predictions = []

    total_samples = min(len(dataset), max_samples)

    # === NEW: throughput counters ===
    total_generate_time_s = 0.0
    total_generated_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size), desc=f"Evaluating {model_name}"):
            batch_data = dataset[i:min(i + batch_size, total_samples)]

            prompts = []
            ground_truths = []

            for j in range(len(batch_data['question'])):
                context = batch_data['context'][j]
                question = batch_data['question'][j]
                answers = batch_data['answers'][j]

                prompt = f"{tokenizer.bos_token} Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                prompts.append(prompt)

                ground_truth = answers['text'][0] if answers['text'] else ""
                ground_truths.append(ground_truth)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # ORT generate는 torch.Tensor를 CPU로 받아도 내부에서 처리함.
            # 생성 시간만 정확히 재자.
            input_len = int(inputs["input_ids"].shape[-1])

            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            t1 = time.perf_counter()

            batch_time = (t1 - t0)
            total_generate_time_s += batch_time

            # === tokens generated ===
            out_len = int(outputs.shape[-1])
            new_tokens_this_batch_total = (out_len - input_len) * outputs.shape[0]
            total_generated_tokens += new_tokens_this_batch_total

            for j, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                answer = extract_answer_from_generation(generated_text, prompts[j])

                f1 = compute_f1(answer, ground_truths[j])
                exact = compute_exact(answer, ground_truths[j])

                f1_scores.append(f1)
                exact_scores.append(exact)

                predictions.append({
                    'context': batch_data['context'][j][:100] + "...",
                    'question': batch_data['question'][j],
                    'prediction': answer,
                    'ground_truth': ground_truths[j],
                    'f1': f1,
                    'exact_match': exact
                })

                if len(predictions) <= 3:
                    logger.info(f"\nExample {len(predictions)}:")
                    logger.info(f"  Question: {batch_data['question'][j]}")
                    logger.info(f"  Predicted: {answer}")
                    logger.info(f"  Ground Truth: {ground_truths[j]}")
                    logger.info(f"  F1: {f1:.3f}, Exact: {exact}")

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    avg_exact = float(np.mean(exact_scores)) if exact_scores else 0.0

    tokens_per_sec = (total_generated_tokens / total_generate_time_s) if total_generate_time_s > 0 else 0.0
    ms_per_token = (1000.0 * total_generate_time_s / max(total_generated_tokens, 1))


    results = {
        'model_name': model_name,
        'num_samples': total_samples,
        'average_f1': avg_f1,
        'average_exact_match': avg_exact,
        'generated_tokens': int(total_generated_tokens),
        'generate_time_seconds': total_generate_time_s,
        'tokens_per_second': tokens_per_sec,
        'ms_per_token': ms_per_token,
    }

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Average F1 Score      : {avg_f1:.4f}")
    logger.info(f"  Average Exact Match   : {avg_exact:.4f}")
    logger.info(f"  Generated tokens      : {total_generated_tokens}")
    logger.info(f"  Generate time (s)     : {total_generate_time_s:.3f}")
    logger.info(f"  Throughput (tok/s)    : {tokens_per_sec:.2f}")
    logger.info(f"  Latency (ms/token)    : {ms_per_token:.2f}")

    return results, predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on SQuAD dataset')
    parser.add_argument('--path', type=str,
                       default='./compressed_llama_squad_sft_kd_selective/final_model',
                       help='Path to trained compressed model (Optimum-ONNX dir)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index for NVML tracking')  
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dtype = torch.float16
    logger.info(f"Using dtype: {dtype}")

    try:
        logger.info("Loading SQuAD validation dataset...")
        dataset = load_dataset("squad", split="validation")

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.path)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # --- [ADD] 현재 args.path에 있는 ONNX 가중치 총 용량 로깅 ---
        onnx_file = _pick_onnx_file(args.path)
        init_bytes = _sum_initializer_bytes(onnx_file)
        logger.info(
            f"[ONNX Weights] file={onnx_file} | "
            f"initializers={init_bytes/1024**2:.2f} MB ({init_bytes/1024**3:.2f} GB)"
        )
        
        # ORT model (Optimum 포맷)
        model = ORTModelForCausalLM.from_pretrained(args.path, provider="CUDAExecutionProvider")

            # for i in range(5):
        trained_results, trained_predictions = evaluate_model(
            model,
            tokenizer,
            dataset,
            model_name="Trained Compressed Model",
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
    

        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)

        comparison = {
            "Trained Compressed Model": {
                "F1 Score": trained_results['average_f1'],
                "Exact Match": trained_results['average_exact_match'],
                "Generated tokens": trained_results['generated_tokens'],
                "Generate time (s)": trained_results['generate_time_seconds'],
                "Throughput (tok/s)": trained_results['tokens_per_second'],
                "Latency (ms/token)": trained_results['ms_per_token'],
            }
        }

        for model_name, metrics in comparison.items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    logger.info(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")
                else:
                    logger.info(f"  {metric_name}: {value}")

        logger.info(f"\nSaving results to {args.output_dir}")
        
        _path = args.path.split("/")[-1]
        
        with open(f"{args.output_dir}/{_path}_evaluation_summary_000.json", 'w') as f:
            results_dict = {
                'trained_results': trained_results,
                'comparison': comparison
            }
            json.dump(results_dict, f, indent=2)

        with open(f"{args.output_dir}/{_path}_sample_predictions.json", 'w') as f:
            pred_dict = {
                'trained_predictions': trained_predictions[:10]
            }
            json.dump(pred_dict, f, indent=2)

        logger.info("\n✅ Evaluation complete! Check the evaluation_results directory for detailed results.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
