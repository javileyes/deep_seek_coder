from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def generate_long_chat(historico, input_text, max_additional_tokens=2000, stop=["<|EOT|>"]):

    # if stop is None:
    #     stop = [eos_token_id]

    prompt = f"### Instruction:\n{input_text}\n### Response:\n"
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) # para streamear el output pero sin repetir el prompt ni el contexto anterior. 

    final_prompt = historico + "\n" + prompt
 
    inputs = tokenizer(final_prompt, return_tensors="pt", add_special_tokens=False)

    model_inputs = inputs.to(model.device)      # .to("cuda")
    outputs = model.generate(**model_inputs,
                             streamer=streamer,
                             max_new_tokens=max_additional_tokens,
                            #  max_length=max_length,
                             temperature=0.1,
                             pad_token_id = 3200,
                             eos_token_id=32021,
                             do_sample=True)

    # Decodificar el tensor de salida a una cadena de texto
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)    
    
    text = final_prompt + decoded_output + "<|EOT|>"
    return text


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True)
#torch_dtype=torch.float16  (half precision) or torch.float32 (single precision)


system_prompt = """
You are an expert AI programming assistant, utilizing the DeepSeek Coder model, and you only answer questions related to computer science.
"""

import sys
import os

if len(sys.argv) > 1:
    system_prompt = sys.argv[1]


historico = system_prompt

while True:
    # read input
    input_text = input("user: ")
    if input_text == "/exit": break
    if input_text == "/historico": 
        print(historico)
        continue
    if input_text == "/clear":
        historico = ""
        continue
    # generate response
    historico = generate_long_chat(historico, input_text=input_text, max_additional_tokens=2048)
    # print response
    # print(salida)
    print(f"\n################################################\n")


# messages = [
#     {'role': 'user', 'content': "write a quick sort algorithm in python."}
# ]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
