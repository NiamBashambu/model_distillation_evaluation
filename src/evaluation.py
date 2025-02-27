import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models and tokenizers
teacher_model_name = "EleutherAI/gpt-neo-1.3B"
distilled_model_path = "./distilled_student_model"

# Teacher model with bfloat16 to save memory
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, torch_dtype=torch.bfloat16
).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Distilled model (assuming same tokenizer unless specified otherwise)
distilled_model = AutoModelForCausalLM.from_pretrained(distilled_model_path).to(device)
distilled_tokenizer = AutoTokenizer.from_pretrained(distilled_model_path)

# Define task configurations for classification benchmarks
task_configs = {
    "SuperGLUE_cb": {
        "prompt_template": "Premise: {premise} Hypothesis: {hypothesis} The relationship is",
        "label_map": {0: "entailment", 1: "contradiction", 2: "neutral"},
        "input_fields": ["premise", "hypothesis"]
    },
    "GLUE_mrpc": {
        "prompt_template": "Sentence1: {sentence1} Sentence2: {sentence2} Are they equivalent or not equivalent? The answer is",
        "label_map": {0: "not equivalent", 1: "equivalent"},
        "input_fields": ["sentence1", "sentence2"]
    },
    "XTREME_XNLI": {
        "prompt_template": "Premise: {premise} Hypothesis: {hypothesis} The relationship is",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "input_fields": ["premise", "hypothesis"]
    },
    "TREC": {
        "prompt_template": "Classify this question: {text} into one of: abbreviation, entity, description, human, location, numeric. The category is",
        "label_map": {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: "numeric"},
        "input_fields": ["text"]
    }
}

# Define benchmarks (classification tasks only)
benchmarks = [
    {"name": "SuperGLUE_cb", "dataset": "super_glue", "config": "cb", "split": "validation"},
    {"name": "GLUE_mrpc", "dataset": "glue", "config": "mrpc", "split": "validation"},
    {"name": "XTREME_XNLI", "dataset": "xtreme", "config": "XNLI", "split": "test"},
    {"name": "TREC", "dataset": "trec", "config": None, "split": "test"}
]

# Evaluation function
def evaluate_model(model, tokenizer, benchmark, task_config):
    """Evaluate a model on a benchmark task using zero-shot classification."""
    # Load dataset
    if benchmark["config"]:
        dataset = load_dataset(benchmark["dataset"], benchmark["config"], split=benchmark["split"])
    else:
        dataset = load_dataset(benchmark["dataset"], split=benchmark["split"])

    if benchmark["name"] == "XTREME_XNLI":
        dataset = dataset.filter(lambda x: x["language"] == "en")

    correct = 0
    total = 0
    for example in dataset:
        # Construct prompt
        input_data = {field: example[field] for field in task_config["input_fields"]}
        prompt = task_config["prompt_template"].format(**input_data)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Get true label
        true_label = example["label"]

        # Compute loss for each verbalizer
        losses = []
        verbalizers = list(task_config["label_map"].values())
        for verbalizer in verbalizers:
            verbalizer_ids = tokenizer.encode(verbalizer, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(input_ids=prompt_ids, labels=verbalizer_ids)
                loss = outputs.loss.item()  # Negative average log prob
            losses.append(loss)

        # Predict label with smallest loss (highest probability)
        pred_idx = np.argmin(losses)
        pred_label = list(task_config["label_map"].keys())[pred_idx]
        if pred_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Collect results
results_data = []

# Evaluate both models on each benchmark
for benchmark in benchmarks:
    task_name = benchmark["name"]
    if task_name in task_configs:
        task_config = task_configs[task_name]
        print(f"Evaluating {task_name}...")
        
        # Teacher model evaluation
        teacher_accuracy = evaluate_model(teacher_model, teacher_tokenizer, benchmark, task_config)
        
        # Distilled model evaluation
        distilled_accuracy = evaluate_model(distilled_model, distilled_tokenizer, benchmark, task_config)
        
        # Store results
        results_data.append({
            "Workload": task_name,
            "Original Model Accuracy": teacher_accuracy,
            "Distilled Model Accuracy": distilled_accuracy,
            "Remarks": "Zero-shot classification performance"
        })
        print(f"{task_name} - Teacher Accuracy: {teacher_accuracy:.4f}, Distilled Accuracy: {distilled_accuracy:.4f}")

# Save results to CSV
df = pd.DataFrame(results_data)
df.to_csv("benchmark_results.csv", index=False)
print("Evaluation completed. Results saved to 'benchmark_results.csv'")