from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch

model_path = "./phi2-local"

# ✅ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cpu")

#model = model.to(torch.float32)
print("✅ phi-2 model loaded in float16 on CPU!")

# ✅ Load dataset
dataset = load_dataset("json", data_files="IncidentNotes.jsonl", split="train")

# ✅ Convert to 'text' field if not present
if "text" not in dataset.column_names:
    def combine_fields(example):
        return {
            "text": f"{example['instruction']}\n\n{example['input']}\n\n### Report:\n{example['output']}"
        }

    dataset = dataset.map(combine_fields)

# ✅ Training config
training_args = TrainingArguments(
    output_dir="./phi2-finetuned-reportgen",
    per_device_train_batch_size=1,
    #max_seq_length=256,
    dataloader_pin_memory=False,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch"
)

# ✅ Trainer setup
trainer = SFTTrainer(
    output_dir="./phi2-finetuned-reportgen",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    model=model,
    train_dataset=dataset,
    #tokenizer=tokenizer,
    args=training_args,
    #dataset_text_field="text",
    #max_seq_length=512
)

# ✅ Train
trainer.train(resume_from_checkpoint=True)

trainer.model.save_pretrained("./phi2-finetuned-reportgen")
tokenizer.save_pretrained("./phi2-finetuned-reportgen")
