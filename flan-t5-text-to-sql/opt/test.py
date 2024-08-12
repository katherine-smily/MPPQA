from transformers import AutoTokenizer, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("/home/cike/bihan/projects/flan-t5-text-to-sql/opt/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("/home/cike/bihan/projects/flan-t5-text-to-sql/opt/opt-350m")
# print(model.config.max_position_embeddings)
prompt = "please answer the question, don't explain. Question: Is cat animal?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
a=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(a)
# "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."