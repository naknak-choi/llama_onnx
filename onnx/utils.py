from __future__ import annotations
import argparse, json, os, sys, warnings, gc
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from contextlib import contextmanager

warnings.filterwarnings("ignore", message="TracerWarning*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_parameters(model, mlp_key="mlp"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlp = sum(p.numel() for n, p in model.named_parameters() if mlp_key in n)
    return {"total": total, "trainable": trainable, "mlp": mlp}


# ---------------- Compressed MLP ----------------
class CompressedMLP(nn.Module):
    """압축된 MLP 구조"""
    def __init__(self, original_gate_proj, original_up_proj, original_down_proj, config):
        super().__init__()
        
        self.compress_gate = config.get("compress_gate", False)
        self.compress_up = config.get("compress_up", False) 
        self.compress_down = config.get("compress_down", False)
        
        rank_ratio = config.get("rank_ratio", 0.5)
        
        hidden_dim, input_dim = original_gate_proj.weight.shape
        output_dim = original_down_proj.weight.shape[0]
        
        # Create compressed structures
        rank_gate = int(input_dim * rank_ratio)
        self.gate_A = nn.Linear(input_dim, rank_gate, bias=False)
        self.gate_B = nn.Linear(rank_gate, hidden_dim, bias=False)
        
        rank_up = int(input_dim * rank_ratio)
        self.up_A = nn.Linear(input_dim, rank_up, bias=False)
        self.up_B = nn.Linear(rank_up, hidden_dim, bias=False)
        
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        up_output = self.up_B(self.up_A(x))
        gate_output = self.gate_B(self.gate_A(x))
        gate_output = self.act_fn(gate_output)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        return output

# ---------------- Model Loading ----------------
def load_compression_artifacts(model_dir: Path):
    """압축 모델 아티팩트 로드"""
    config_path = model_dir / "compression_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"compression_config.json not found in {model_dir}")
    
    with open(config_path, 'r') as f:
        compression_config = json.load(f)
    
    return compression_config

def load_compressed_model(model_path: Path, device: str = 'cuda'):
    """메모리 효율적으로 압축 모델 로드"""
    print("Loading compressed model...")
    
    # 1. 압축 설정 로드
    compression_config = load_compression_artifacts(model_path)
    print(f"Loaded compression config: {compression_config}")
    
    # 2. 모델 설정만 로드
    config = AutoConfig.from_pretrained(model_path)
    
    if hasattr(config, "attn_implementation"):
        print(f"Original attn_implementation: {config.attn_implementation}")
        print("Setting config.attn_implementation to 'eager' for ONNX compatibility.")
        config.attn_implementation = "eager"

    # 3. CPU에서 빈 모델 구조 생성 (수정된 config 사용)
    with torch.device('cpu'):
        model = AutoModelForCausalLM.from_config(config)
    
    # 4. MLP 레이어를 압축 버전으로 교체
    num_layers = len(model.model.layers)
    layer_indices = compression_config.get('layer_indices', list(range(num_layers)))
    
    if layer_indices is None or layer_indices == "None":
        layer_indices = list(range(num_layers))
    
    for layer_idx in layer_indices:
        if layer_idx < num_layers:
            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'mlp'):
                original_mlp = layer.mlp
                
                # 압축된 MLP 생성
                layer.mlp = CompressedMLP(
                    original_mlp.gate_proj,
                    original_mlp.up_proj,
                    original_mlp.down_proj,
                    compression_config
                )
                
                del original_mlp
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import glob
    from safetensors.torch import load_file
    
    state_dict = {}
    
    # safetensors 우선 시도
    safetensor_files = glob.glob(str(model_path / "*.safetensors"))
    if safetensor_files:
        for file in safetensor_files:
            state_dict.update(load_file(file))
    else:
        # pytorch bins 시도
        bin_files = glob.glob(str(model_path / "*.bin"))
        for file in bin_files:
            state_dict.update(torch.load(file, map_location='cpu'))
    
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(torch.float16)

    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model