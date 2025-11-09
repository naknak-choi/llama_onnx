#!/usr/bin/env python3
import os, json, argparse, statistics as stats
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import pynvml

# ---------- utils ----------
def pick_onnx(model_dir: Path) -> str:
    for n in ("decoder_model.onnx", "decoder_with_past_model.onnx", "model.onnx"):
        p = model_dir / n
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No ONNX file under {model_dir}")

def resolve_nvml_index(visible_idx=0):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return visible_idx
    lst = [int(x) for x in cvd.split(",") if x != ""]
    if not lst:
        return visible_idx
    if visible_idx < 0 or visible_idx >= len(lst):
        visible_idx = 0
    return lst[visible_idx]

def nvml_used_bytes(phys_idx=0) -> int:
    h = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used

def build_position_ids(attn_mask: np.ndarray) -> np.ndarray:
    pos = np.cumsum(attn_mask, axis=1) - 1
    pos = np.maximum(pos, 0)
    return pos.astype(np.int64)

def build_dummy_past(cfg: dict, batch: int, past_len: int, dtype=np.float32):
    L   = int(cfg["num_hidden_layers"])
    H   = int(cfg["num_attention_heads"])
    KV  = int(cfg.get("num_key_value_heads", H))
    hd  = int(cfg["hidden_size"]) // H
    past = []
    for _ in range(L):
        k = np.zeros((batch, KV, past_len, hd), dtype=dtype)
        v = np.zeros((batch, KV, past_len, hd), dtype=dtype)
        past.append((k, v))
    return past

def make_feeds(sess: ort.InferenceSession, tok_dir: Path, prompt: str, max_len: int):
    input_names = [i.name for i in sess.get_inputs()]
    need_posids = "position_ids" in input_names
    need_past   = any(n.startswith(("past_key_values", "past_key_")) for n in input_names)

    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    tok.pad_token = tok.eos_token
    
    # 재현성↑: 길이를 고정해서 패딩
    enc = tok(prompt, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    ids  = enc["input_ids"].astype(np.int64)
    mask = enc.get("attention_mask", np.ones_like(ids, dtype=np.int64)).astype(np.int64)

    feeds = {}
    if "input_ids" in input_names: feeds["input_ids"] = ids
    if "attention_mask" in input_names: feeds["attention_mask"] = mask
    if need_posids:
        feeds["position_ids"] = build_position_ids(mask)
    if need_past:
        cfg = json.load(open(Path(tok_dir) / "config.json", "r", encoding="utf-8"))
        dummy = build_dummy_past(cfg, batch=ids.shape[0], past_len=0, dtype=np.float16)
        for i, (k, v) in enumerate(dummy):
            kname = f"past_key_values.{i}.key" if f"past_key_values.{i}.key" in input_names else f"past_key_{i}"
            vname = f"past_key_values.{i}.value" if f"past_key_values.{i}.value" in input_names else f"past_value_{i}"
            feeds[kname] = k; feeds[vname] = v
    return feeds

# ---------- one trial ----------
def run_one_trial(model_dir: Path, visible_gpu_idx: int, prompt: str, max_length: int,
                  arena_same_as_requested: bool, warmups: int, gpu_mem_limit_gb: float | None):
    phys = resolve_nvml_index(visible_gpu_idx)
    onnx_path = pick_onnx(model_dir)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_cfg = {}
    if arena_same_as_requested:
        cuda_cfg["arena_extend_strategy"] = "kSameAsRequested"
    if gpu_mem_limit_gb is not None:
        cuda_cfg["gpu_mem_limit"] = str(int(gpu_mem_limit_gb * (1024**3)))

    providers = [("CUDAExecutionProvider", cuda_cfg), "CPUExecutionProvider"]

    m0 = nvml_used_bytes(phys) / (1024**3)

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    m1 = nvml_used_bytes(phys) / (1024**3)
    d_weights = m1 - m0

    feeds = make_feeds(sess, model_dir, prompt, max_length)

    # warmup (측정 X)
    for _ in range(max(warmups, 0)):
        _ = sess.run(None, feeds)

    # measured run (prefill 1회)
    _ = sess.run(None, feeds)
    m2 = nvml_used_bytes(phys) / (1024**3)
    d_run = m2 - m1

    # 정리 (동일 프로세스 내 잔류 최소화)
    del sess
    import gc; gc.collect()

    return {
        "before": m0,
        "after_session": m1,
        "after_prefill": m2,
        "delta_weights": d_weights,
        "delta_run": d_run,
        "providers": ort.get_available_providers()
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Optimum-ONNX dir (contains decoder_model.onnx or decoder_with_past_model.onnx)")
    ap.add_argument("--gpu", type=int, default=0, help="visible GPU index (respecting CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--prompt", type=str, default="hello")
    ap.add_argument("--max_length", type=int, default=128, help="fixed input token length (padding/truncation)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--arena_same", action="store_true", help="use arena_extend_strategy=kSameAsRequested")
    ap.add_argument("--gpu_mem_limit_gb", type=float, default=None, help="optional hard cap, e.g., 7.5")
    args = ap.parse_args()

    model_dir = Path(args.path)

    print(f"[CUDA_VISIBLE_DEVICES]={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Model dir] {model_dir}")
    print(f"[Settings] gpu={args.gpu}, max_length={args.max_length}, trials={args.trials}, warmups={args.warmups}, arena_same={args.arena_same}, gpu_mem_limit_gb={args.gpu_mem_limit_gb}")

    pynvml.nvmlInit()
    try:
        results = []
        for t in range(args.trials):
            r = run_one_trial(
                model_dir=model_dir,
                visible_gpu_idx=args.gpu,
                prompt=args.prompt,
                max_length=args.max_length,
                arena_same_as_requested=args.arena_same,
                warmups=args.warmups,
                gpu_mem_limit_gb=args.gpu_mem_limit_gb,
            )
            results.append(r)
            print(f"[trial {t}] before={r['before']:.3f} GB | after_session={r['after_session']:.3f} GB | after_prefill={r['after_prefill']:.3f} GB | Δweights={r['delta_weights']:.3f} GB | Δrun={r['delta_run']:.3f} GB")

        def agg(key):
            xs = [x[key] for x in results]
            return {
                "avg": stats.mean(xs),
                "std": stats.pstdev(xs) if len(xs) > 1 else 0.0,
                "min": min(xs),
                "max": max(xs)
            }

        agg_w = agg("delta_weights")
        agg_r = agg("delta_run")
        print("\n=== SUMMARY ===")
        print(f"Δweights (avg±std, min~max): {agg_w['avg']:.3f}±{agg_w['std']:.3f}  {agg_w['min']:.3f}~{agg_w['max']:.3f} GB")
        print(f"Δrun     (avg±std, min~max): {agg_r['avg']:.3f}±{agg_r['std']:.3f}  {agg_r['min']:.3f}~{agg_r['max']:.3f} GB")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
