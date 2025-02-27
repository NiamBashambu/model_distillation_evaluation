import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

# Load Student Model & Tokenizer (Properly Aligned)
student_model_name = "distilgpt2"  # Smaller variant for testing
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.float16)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, padding_side="left")
student_tokenizer.pad_token = student_tokenizer.eos_token  

teacher_model_name = "EleutherAI/gpt-neo-1.3B"  
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.float16)
teacher_model.eval()

# Distillation Loss (Combined KL + CE)
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # KL Divergence
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    
    # Cross-Entropy
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
    return alpha * kl_loss + (1 - alpha) * ce_loss

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):        
        with torch.no_grad():
            teacher_outputs = teacher_model(**{k: v.to(teacher_model.device) for k, v in inputs.items()})
        
        # Student Forward
        student_outputs = model(**inputs)
        
        # Compute Loss
        loss = distillation_loss(
            student_outputs.logits, 
            teacher_outputs.logits.to(student_outputs.logits.device), 
            inputs["labels"]
        )
        return (loss, student_outputs) if return_outputs else loss

# Dataset (Tokenized with Student's Tokenizer)
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1%]")  # Subset for testing
def tokenize_function(examples):
    tokenized = student_tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    # Create labels by shifting input_ids
    labels = tokenized["input_ids"].copy()
    labels = [label if label != student_tokenizer.pad_token_id else -100 for label in labels]  # Ignore padding tokens in loss computation
    
    tokenized["labels"] = labels
    
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# Training Arguments
training_args = TrainingArguments(
    output_dir="./distill_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="no",
)

# Trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=student_tokenizer,
    data_collator=None,  

)

trainer.train()

student_model.save_pretrained("./distilled_student_model")
student_tokenizer.save_pretrained("./distilled_student_model")
     