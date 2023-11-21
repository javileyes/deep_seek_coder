# cuantificacion de 8 bits (gracias a la libreria de transformers y parámetro load_in_8bit=True)
# Para utilizar la cuantificación de 8 bits en modelos de Transformers, puedes usar el argumento "load_in_8bit=True" en el método from_pretrained. Esto te permite cargar un modelo reduciendo aproximadamente a la mitad los requisitos de memoria. Este método es compatible con modelos que admiten la carga con la biblioteca Accelerate y que contienen capas "torch.nn.Linear"​

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', load_in_8bit=True, trust_remote_code=True)

# messages = [
#     {'role': 'user', 'content': "write a quick sort algorithm in python."}
# ]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))



from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True)

messages = [
    {'role': 'user', 'content': "write a quick sort algorithm in python."}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))




