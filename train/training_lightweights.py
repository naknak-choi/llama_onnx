import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import os
import json
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressedMLP(nn.Module):
    """SVD-compressed MLP module with selective compression and reconstruction error minimization"""
    
    def __init__(self, original_gate_proj, original_up_proj, original_down_proj, 
                 rank_ratio=0.25, compress_gate=True, compress_up=True, compress_down=True,
                 init_scale=1.0, use_residual=False):
        """
        Initialize compressed MLP with selective compression.
        
        Args:
            original_gate_proj: Original gate projection layer
            original_up_proj: Original up projection layer
            original_down_proj: Original down projection layer
            rank_ratio: Compression ratio (fraction of original rank to keep)
            compress_gate: Whether to compress gate_proj
            compress_up: Whether to compress up_proj
            compress_down: Whether to compress down_proj
            init_scale: Scale factor for initialization
            use_residual: Whether to add residual connection for compressed layers
        """
        super().__init__()
        
        # Store compression flags
        self.compress_gate = compress_gate
        self.compress_up = compress_up
        self.compress_down = compress_down
        self.use_residual = use_residual
        
        # Get dimensions
        hidden_dim, input_dim = original_gate_proj.weight.shape
        output_dim = original_down_proj.weight.shape[0]
        
        compression_info = []
        reconstruction_errors = []
        
        # Handle gate_proj
        if compress_gate:
            self.rank_gate = max(int(min(hidden_dim, input_dim) * rank_ratio), 1)
            compression_info.append(f"gate_proj: {hidden_dim}x{input_dim} -> rank {self.rank_gate}")
            
            # Compress gate_proj using SVD with better initialization
            with torch.no_grad():
                weight_fp32 = original_gate_proj.weight.data.float()
                U_gate, S_gate, V_gate = torch.svd_lowrank(weight_fp32, q=self.rank_gate)
                
                # Create compressed layers
                self.gate_A = nn.Linear(input_dim, self.rank_gate, bias=False, dtype=original_gate_proj.weight.dtype)
                self.gate_B = nn.Linear(self.rank_gate, hidden_dim, bias=False, dtype=original_gate_proj.weight.dtype)
                
                # Initialize with scaled SVD factors
                self.gate_A.weight.data = (V_gate.T * init_scale).to(original_gate_proj.weight.dtype)
                self.gate_B.weight.data = (U_gate @ torch.diag(S_gate) * init_scale).to(original_gate_proj.weight.dtype)
                
                # Calculate reconstruction error
                reconstructed = U_gate @ torch.diag(S_gate) @ V_gate.T
                error = torch.norm(weight_fp32 - reconstructed) / torch.norm(weight_fp32)
                reconstruction_errors.append(f"gate_proj: {error:.4f}")
                
                # Optional: store residual for fine-tuning
                if self.use_residual:
                    self.gate_residual = nn.Parameter(
                        torch.zeros_like(original_gate_proj.weight.data) * 0.01
                    )
        else:
            # Keep original gate_proj
            self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False, dtype=original_gate_proj.weight.dtype)
            self.gate_proj.weight.data = original_gate_proj.weight.data.clone()
            compression_info.append(f"gate_proj: keeping original {hidden_dim}x{input_dim}")
        
        # Handle up_proj
        if compress_up:
            self.rank_up = max(int(min(hidden_dim, input_dim) * rank_ratio), 1)
            compression_info.append(f"up_proj: {hidden_dim}x{input_dim} -> rank {self.rank_up}")
            
            # Compress up_proj using SVD
            with torch.no_grad():
                weight_fp32 = original_up_proj.weight.data.float()
                U_up, S_up, V_up = torch.svd_lowrank(weight_fp32, q=self.rank_up)
                
                self.up_A = nn.Linear(input_dim, self.rank_up, bias=False, dtype=original_up_proj.weight.dtype)
                self.up_B = nn.Linear(self.rank_up, hidden_dim, bias=False, dtype=original_up_proj.weight.dtype)
                
                self.up_A.weight.data = (V_up.T * init_scale).to(original_up_proj.weight.dtype)
                self.up_B.weight.data = (U_up @ torch.diag(S_up) * init_scale).to(original_up_proj.weight.dtype)
                
                # Calculate reconstruction error
                reconstructed = U_up @ torch.diag(S_up) @ V_up.T
                error = torch.norm(weight_fp32 - reconstructed) / torch.norm(weight_fp32)
                reconstruction_errors.append(f"up_proj: {error:.4f}")
                
                if self.use_residual:
                    self.up_residual = nn.Parameter(
                        torch.zeros_like(original_up_proj.weight.data) * 0.01
                    )
        else:
            # Keep original up_proj
            self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False, dtype=original_up_proj.weight.dtype)
            self.up_proj.weight.data = original_up_proj.weight.data.clone()
            compression_info.append(f"up_proj: keeping original {hidden_dim}x{input_dim}")
        
        # Handle down_proj - CRITICAL FOR OUTPUT QUALITY
        if compress_down:
            self.rank_down = max(int(min(output_dim, hidden_dim) * rank_ratio), 1)
            compression_info.append(f"down_proj: {output_dim}x{hidden_dim} -> rank {self.rank_down}")
            
            # Compress down_proj using SVD with special care
            with torch.no_grad():
                weight_fp32 = original_down_proj.weight.data.float()
                
                # Use higher precision SVD for down_proj
                U_down, S_down, V_down = torch.svd_lowrank(weight_fp32, q=self.rank_down)
                
                self.down_A = nn.Linear(hidden_dim, self.rank_down, bias=False, dtype=original_down_proj.weight.dtype)
                self.down_B = nn.Linear(self.rank_down, output_dim, bias=False, dtype=original_down_proj.weight.dtype)
                
                # Careful initialization for down_proj (affects output directly)
                self.down_A.weight.data = (V_down.T * init_scale).to(original_down_proj.weight.dtype)
                self.down_B.weight.data = (U_down @ torch.diag(S_down) * init_scale).to(original_down_proj.weight.dtype)
                
                # Calculate reconstruction error
                reconstructed = U_down @ torch.diag(S_down) @ V_down.T
                error = torch.norm(weight_fp32 - reconstructed) / torch.norm(weight_fp32)
                reconstruction_errors.append(f"down_proj: {error:.4f}")
                
                if self.use_residual:
                    self.down_residual = nn.Parameter(
                        torch.zeros_like(original_down_proj.weight.data) * 0.01
                    )
        else:
            # Keep original down_proj
            self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False, dtype=original_down_proj.weight.dtype)
            self.down_proj.weight.data = original_down_proj.weight.data.clone()
            compression_info.append(f"down_proj: keeping original {output_dim}x{hidden_dim}")
        
        # Store activation function
        self.act_fn = nn.SiLU()
        
        # Log compression info
        logger.info("MLP Compression: " + ", ".join(compression_info))
        if reconstruction_errors:
            logger.info("Reconstruction errors: " + ", ".join(reconstruction_errors))
    
    def forward(self, x):
        # Gate projection
        if self.compress_gate:
            gate_output = self.gate_B(self.gate_A(x))
            if self.use_residual and hasattr(self, 'gate_residual'):
                gate_output = gate_output + F.linear(x, self.gate_residual)
        else:
            gate_output = self.gate_proj(x)
        gate_output = self.act_fn(gate_output)
        
        # Up projection
        if self.compress_up:
            up_output = self.up_B(self.up_A(x))
            if self.use_residual and hasattr(self, 'up_residual'):
                up_output = up_output + F.linear(x, self.up_residual)
        else:
            up_output = self.up_proj(x)
        
        # Element-wise multiplication (like in original Llama)
        intermediate = gate_output * up_output
        
        # Down projection - CRITICAL PATH
        if self.compress_down:
            output = self.down_B(self.down_A(intermediate))
            if self.use_residual and hasattr(self, 'down_residual'):
                output = output + F.linear(intermediate, self.down_residual)
        else:
            output = self.down_proj(intermediate)
        
        return output
    
    def count_parameters(self):
        """Count the number of parameters in this module"""
        return sum(p.numel() for p in self.parameters())

def test_compression_quality(model, tokenizer, test_prompts=None):
    """Test the quality of compressed model outputs"""
    
    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "The sky is",
            "Hello, my name is",
            "The quick brown fox",
        ]
    
    model.eval()
    results = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(f"{prompt} -> {generated[len(prompt):]}")
    
    return results

def compress_model_mlp_safe(model, rank_ratio=0.25, compress_gate=True, compress_up=True, 
                            compress_down=True, layer_indices=None, test_compression=True,
                            tokenizer=None):
    """
    Safely compress MLP modules with quality checks.
    """
    
    total_original_params = 0
    total_compressed_params = 0
    
    # Determine which layers to compress
    num_layers = len(model.model.layers)
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    
    logger.info(f"Compression settings:")
    logger.info(f"  - Rank ratio: {rank_ratio}")
    logger.info(f"  - Compress gate_proj: {compress_gate}")
    logger.info(f"  - Compress up_proj: {compress_up}")
    logger.info(f"  - Compress down_proj: {compress_down}")
    logger.info(f"  - Layers to compress: {layer_indices if len(layer_indices) < num_layers else 'All'}")
    
    # Test before compression if requested
    if test_compression and tokenizer is not None:
        logger.info("\nTesting model BEFORE compression:")
        before_results = test_compression_quality(model, tokenizer)
        for result in before_results:
            logger.info(f"  {result}")
    
    # Iterate through specified layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        if layer_idx not in layer_indices:
            logger.info(f"Layer {layer_idx}: Skipping (not in compression list)")
            continue
        
        if hasattr(layer, 'mlp'):
            original_mlp = layer.mlp
            
            # Count original parameters
            orig_params = sum(p.numel() for p in original_mlp.parameters())
            total_original_params += orig_params
            
            # Create compressed MLP with better initialization
            compressed_mlp = CompressedMLP(
                original_mlp.gate_proj,
                original_mlp.up_proj,
                original_mlp.down_proj,
                rank_ratio=rank_ratio,
                compress_gate=compress_gate,
                compress_up=compress_up,
                compress_down=compress_down,
                init_scale=1.0,  # Keep original scale
                use_residual=False  # Can be enabled for fine-tuning
            )
            
            # Count compressed parameters
            comp_params = compressed_mlp.count_parameters()
            total_compressed_params += comp_params
            
            # Replace the MLP
            layer.mlp = compressed_mlp
            
            compression_ratio = comp_params/orig_params if orig_params > 0 else 0
            logger.info(f"Layer {layer_idx}: {orig_params:,} -> {comp_params:,} params ({compression_ratio:.2%})")
    
    # Test after compression if requested
    if test_compression and tokenizer is not None:
        logger.info("\nTesting model AFTER compression:")
        after_results = test_compression_quality(model, tokenizer)
        for result in after_results:
            logger.info(f"  {result}")
    
    if total_original_params > 0:
        logger.info(f"\nTotal MLP compression: {total_original_params:,} -> {total_compressed_params:,} params")
        logger.info(f"Overall compression ratio: {total_compressed_params/total_original_params:.2%}")
    
    return model

def freeze_non_mlp_parameters(model):
    """Freeze all parameters except the compressed MLP layers"""
    
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, unfreeze only the compressed MLP parameters
    trainable_params = 0
    frozen_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, CompressedMLP):
            for param in module.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Status:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Verify that we have trainable parameters
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Check CompressedMLP initialization.")

def prepare_squad_dataset(tokenizer, max_length=512):
    """Prepare SQuAD dataset for Supervised Fine-Tuning (SFT)"""
    
    dataset = load_dataset("squad")
    
    PROMPT_TEMPLATE = "{bos_token} Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    def format_squad_examples(examples):
        """Format SQuAD examples for training"""
        formatted_inputs = []
        formatted_targets = []
        
        for question, context, answers in zip(examples["question"], examples["context"], examples["answers"]):
            # Create prompt
            prompt = PROMPT_TEMPLATE.format(bos_token=tokenizer.bos_token, context=context, question=question)
            
            # Get answer
            answer_text = answers["text"][0] if answers["text"] else "No answer"
            
            # Full text for training
            full_text = f"{prompt} {answer_text} {tokenizer.eos_token}"
            
            formatted_inputs.append(full_text)
            formatted_targets.append(len(prompt))
        
        return {
            "text": formatted_inputs,
            "prompt_length": formatted_targets
        }

    def tokenize_and_create_labels(examples):
        """Tokenize and create labels with proper masking"""
        
        # Tokenize with consistent settings
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=False,
        )
        
        labels = []
        
        for i in range(len(examples["text"])):
            # Get the prompt length in characters
            prompt_char_length = examples["prompt_length"][i]
            
            # Tokenize just the prompt to get token count
            prompt_text = examples["text"][i][:prompt_char_length]
            prompt_encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            prompt_token_length = len(prompt_encoding["input_ids"])
            
            # Create labels: mask prompt, keep answer
            sample_labels = model_inputs["input_ids"][i].copy()
            
            # Mask the prompt tokens
            for j in range(min(prompt_token_length, len(sample_labels))):
                sample_labels[j] = -100
            
            # Debug first sample
            if i == 0:
                answer_tokens = sum(1 for token in sample_labels if token != -100)
                print(f"\n=== Tokenization Debug ===")
                print(f"Total tokens: {len(sample_labels)}")
                print(f"Prompt tokens (masked): {prompt_token_length}")
                print(f"Answer tokens (training): {answer_tokens}")
                
                if answer_tokens > 0:
                    # Decode answer part to verify
                    answer_token_ids = [tid for tid in sample_labels if tid != -100]
                    if answer_token_ids:
                        decoded_answer = tokenizer.decode(answer_token_ids[:20])
                        print(f"Answer preview: {decoded_answer}")
            
            labels.append(sample_labels)
        
        model_inputs["labels"] = labels
        
        # Remove intermediate fields
        if "prompt_length" in model_inputs:
            del model_inputs["prompt_length"]
        
        return model_inputs

    small_validation_set = dataset["validation"].select(range(1000))
    
    logger.info("Formatting SQuAD dataset...")
    formatted_train = dataset["train"].map(
        format_squad_examples,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    formatted_validation = small_validation_set.map(
        format_squad_examples,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    logger.info("Tokenizing and creating labels...")
    tokenized_train = formatted_train.map(
        tokenize_and_create_labels,
        batched=True,
        remove_columns=["text", "prompt_length"],
        num_proc=4
    )
    tokenized_validation = formatted_validation.map(
        tokenize_and_create_labels,
        batched=True,
        remove_columns=["text", "prompt_length"],
        num_proc=4
    )
    
    return tokenized_train, tokenized_validation


class DistillationTrainer(Trainer):
    """Custom Trainer for Knowledge Distillation with KL Divergence"""
    
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        if self.teacher_model:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute combined loss with safe parameter handling"""
        
        # Extract labels
        labels = inputs.get("labels")
        if labels is not None:
            labels = inputs.pop("labels")
        
        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Compute SFT loss
        if labels is not None:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            sft_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            sft_loss = student_outputs.loss
        
        # Compute distillation loss if teacher exists
        if self.teacher_model is not None and labels is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Temperature-scaled KL divergence
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            valid_mask = (shift_labels != -100).float()
            
            student_log_probs = F.log_softmax(shift_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
            
            kl_div = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='none'
            ).sum(dim=-1)
            
            kl_loss = (kl_div * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            kl_loss = kl_loss * (self.temperature ** 2)
            
            total_loss = (1 - self.alpha) * sft_loss + self.alpha * kl_loss
            
            # Log losses periodically
            if self.state.global_step % 10 == 0:
                logger.debug(f"Step {self.state.global_step} - SFT: {sft_loss:.4f}, KL: {kl_loss:.4f}")
        else:
            total_loss = sft_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss


def verify_model_generation(model, tokenizer, prompt="Context: The sky is blue.\n\nQuestion: What color is the sky?\n\nAnswer:"):
    """Quick test to verify model can generate"""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generation test - Input: {prompt}")
    logger.info(f"Generation test - Output: {generated}")
    
    return generated


def main():
    # Configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    rank_ratio = 0.5  # Keep k% of rank for better quality
    output_dir = "./compressed_llama_squad_sft_kd_selective"
    compressed_model_path = "./compressed_llama_initial_sft_selective"
    
    # Selective compression settings
    compress_gate = True  # Keep original gate_proj
    compress_up = True    # Keep original up_proj  
    compress_down = False   # Only compress down_proj
    layer_indices = None   # None means compress all layers
    
    # Distillation parameters
    alpha = 0.5  # Higher alpha for more knowledge distillation when compressing critical layers
    temperature = 3.0  # Higher temperature for softer distributions
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float32
    else:
        dtype = torch.float32
        bf16_supported = False
    
    try:
        # 1. Load tokenizer with proper settings
        logger.info("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 2. Load teacher model
        logger.info("2. Loading teacher model...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        teacher_model.eval()
        
        # 3. Load and compress student model with quality testing
        logger.info("3. Loading and compressing student model...")
        student_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Apply selective compression with quality checks
        student_model = compress_model_mlp_safe(
            student_model, 
            rank_ratio=rank_ratio,
            compress_gate=compress_gate,
            compress_up=compress_up,
            compress_down=compress_down,
            layer_indices=layer_indices,
            test_compression=True,
            tokenizer=tokenizer
        )
        
        # 4. Test generation before training
        logger.info("4. Testing generation before training...")
        logger.info("Student model generation:")
        verify_model_generation(student_model, tokenizer)
        logger.info("Teacher model generation:")
        verify_model_generation(teacher_model, tokenizer)
        
        # 5. Save initial compressed model
        logger.info("5. Saving initial compressed model...")
        os.makedirs(compressed_model_path, exist_ok=True)
        student_model.save_pretrained(compressed_model_path)
        tokenizer.save_pretrained(compressed_model_path)
        
        # Save compression configuration
        compression_config = {
            "rank_ratio": rank_ratio,
            "compress_gate": compress_gate,
            "compress_up": compress_up,
            "compress_down": compress_down,
            "layer_indices": layer_indices,
            "warning": "Model needs fine-tuning after compression to restore quality"
        }
        with open(os.path.join(compressed_model_path, "compression_config.json"), "w") as f:
            json.dump(compression_config, f, indent=2)
        
        # 6. Freeze non-MLP parameters
        logger.info("6. Freezing non-MLP parameters...")
        freeze_non_mlp_parameters(student_model)
        
        # Disable cache for training
        student_model.config.use_cache = False
        teacher_model.config.use_cache = False
        
        # 7. Prepare dataset
        logger.info("7. Preparing SQuAD dataset...")
        train_dataset, val_dataset = prepare_squad_dataset(tokenizer)
        
        # 8. Training arguments - adjusted for better recovery
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            learning_rate=5e-5,  # Slightly higher LR for faster recovery
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=2,  # More epochs for better recovery
            weight_decay=0.01,
            warmup_steps=200,  # More warmup steps
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=False,
            bf16=bf16_supported,
            gradient_checkpointing=False,
            report_to="tensorboard",
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=student_model,
            label_pad_token_id=-100,
            padding=True,
            max_length=512,
        )
        
        # 9. Create trainer with adjusted distillation params
        logger.info("9. Creating DistillationTrainer...")
        trainer = DistillationTrainer(
            model=student_model,
            teacher_model=teacher_model,
            alpha=alpha,
            temperature=temperature,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # 10. Train
        logger.info("10. Starting training...")
        logger.info("NOTE: The model will generate poor outputs initially due to compression.")
        logger.info("      Training with knowledge distillation will recover the quality.")
        trainer.train()
        
        # 11. Save final model
        logger.info("11. Saving final model...")
        final_model_path = f"{output_dir}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Save compression configuration with final model
        with open(os.path.join(final_model_path, "compression_config.json"), "w") as f:
            json.dump(compression_config, f, indent=2)
        
        # 12. Test generation after training
        logger.info("12. Testing generation after training...")
        student_model.config.use_cache = True
        verify_model_generation(student_model, tokenizer)
        
        # Additional quality tests
        logger.info("\n13. Running comprehensive quality tests...")
        test_prompts = [
            "Context: The sky is blue.\n\nQuestion: What color is the sky?\n\nAnswer:",
            "Context: Paris is the capital of France.\n\nQuestion: What is the capital of France?\n\nAnswer:",
            "Context: Water freezes at 0 degrees Celsius.\n\nQuestion: At what temperature does water freeze?\n\nAnswer:",
        ]
        
        for prompt in test_prompts:
            generated = verify_model_generation(student_model, tokenizer, prompt)
        
        logger.info("✅ Training complete!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()