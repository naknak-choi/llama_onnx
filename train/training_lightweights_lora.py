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
from peft import get_peft_model, LoraConfig, TaskType  # LoRA를 위한 import 추가
import os
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# 모델 생성 결과 확인 함수 (기존 코드와 유사)
def verify_model_generation(model, tokenizer, prompt="Context: The sky is blue.\n\nQuestion: What color is the sky?\n\nAnswer:"):
    """모델이 텍스트를 생성하는지 간단히 테스트합니다."""
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
    logger.info(f"생성 테스트 - 입력: {prompt}")
    logger.info(f"생성 테스트 - 출력: {generated}")
    return generated

def main():
    # --- 설정 ---
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir = "./llama_squad_lora" # LoRA 학습 결과 저장 경로
    
    # --- 장치 및 데이터 타입 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 장치: {device}")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    try:
        # 1. 토크나이저 로드
        logger.info("1. 토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left' # 패딩 방향 설정
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # 2. 기본 모델 로드
        logger.info("2. 기본 모델 로딩 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 3. LoRA 설정 및 모델 적용
        logger.info("3. LoRA 설정 적용 중...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            lora_dropout=0.05,
            # Llama 모델의 일반적인 Linear 레이어들을 타겟으로 설정
            target_modules=["gate_proj", "up_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        logger.info("LoRA 적용 후 모델 정보:")
        model.print_trainable_parameters() # 학습 가능한 파라미터 수 출력

        # 학습을 위해 캐시 비활성화
        model.config.use_cache = False
        
        # 4. 데이터셋 준비
        logger.info("4. SQuAD 데이터셋 준비 중...")
        train_dataset, val_dataset = prepare_squad_dataset(tokenizer)
        
        # 5. 학습 인자 (Training Arguments) 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1, # 1 에포크로 변경 (필요시 조정)
            weight_decay=0.01,
            warmup_steps=100,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=False,
            bf16=(dtype == torch.bfloat16),
            report_to="tensorboard",
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding=True,
            max_length=512,
        )
        
        # 6. Trainer 생성 (Hugging Face 기본 Trainer 사용)
        logger.info("6. Trainer 생성 중...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # 7. 학습 시작
        logger.info("7. LoRA 파인튜닝 시작...")
        trainer.train()
        
        # 8. 최종 모델(어댑터) 저장
        logger.info("8. 최종 모델 저장 중...")
        final_model_path = f"{output_dir}/final_model"
        trainer.save_model(final_model_path) # LoRA 어댑터 가중치만 저장됨
        tokenizer.save_pretrained(final_model_path)
        
        # 9. 학습 후 생성 테스트
        logger.info("9. 학습 완료 후 생성 테스트...")
        model.config.use_cache = True
        verify_model_generation(model, tokenizer)
        
        logger.info("✅ LoRA 파인튜닝 완료!")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()