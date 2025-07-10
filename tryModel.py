from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./phi2-finetuned-reportgen")
tokenizer = AutoTokenizer.from_pretrained("./phi2-finetuned-reportgen")

input_text = "Generate a report from incident notes.\n2023-04-04 10:32AM: Firewall alert triggered. User reported lag. DNS lookup failures observed."
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
