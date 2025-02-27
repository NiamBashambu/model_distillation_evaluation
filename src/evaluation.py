import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import pandas as pd

# List of benchmark tasks and datasets
benchmarks = [
    ("SuperGLUE", "super_glue", "cb"),
    ("GLUE", "glue", "mrpc"),
    ("XTREME", "xtreme", "XTRA_XX"),
    ("SQuAD", "squad", "train"),
    ("Conll-03", "conll2003", "test"),
    ("WMT", "wmt14", "train"),
    ("TREC", "trec", "train")
]

# Load models
teacher_model_name = "EleutherAI/gpt-neo-1.3B"
distilled_model_path = "./distilled_student_model"

teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16, device_map="auto")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

distilled_model = AutoModelForCausalLM.from_pretrained(distilled_model_path)
distilled_tokenizer = AutoTokenizer.from_pretrained(distilled_model_path)

# Function to evaluate a model on a specific benchmark task
def evaluate_model(model, tokenizer, benchmark_name, dataset_name, split_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split_name)
    
    # Setup evaluation pipeline (text classification for most tasks)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    # Evaluate on the dataset - error handling if not named the correct thing
    if "sentence1" in dataset.column_names:
        input_data = dataset["sentence1"]
    elif "text" in dataset.column_names:
        input_data = dataset["text"]
    else:
        input_data = dataset[dataset.column_names[0]]

    results = classifier(input_data)

    # Extract predicted labels and true labels
    pred_labels = [result['label'] for result in results]
    true_labels = dataset['label']
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    return accuracy

# Prepare to collect results
results_data = []

# Iterate over all benchmark tasks and models
for benchmark_name, dataset_name, split_name in benchmarks:
    # Evaluate original model
    original_accuracy = evaluate_model(teacher_model, teacher_tokenizer, benchmark_name, dataset_name, split_name)
    
    # Evaluate distilled model
    distilled_accuracy = evaluate_model(distilled_model, distilled_tokenizer, benchmark_name, dataset_name, split_name)
    
    # Add results to list
    results_data.append({
        "Workload": benchmark_name,
        "Original Model Accuracy": original_accuracy,
        "Distilled Model Accuracy": distilled_accuracy,
        "Remarks": "Performance comparison"
    })

# Convert results to a pandas DataFrame
df = pd.DataFrame(results_data)

# Save the results to a CSV file
df.to_csv("benchmark_results.csv", index=False)

print("Evaluation completed. Results saved to 'benchmark_results.csv'")
