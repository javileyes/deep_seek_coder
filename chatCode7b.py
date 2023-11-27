from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def ajustar_contexto(texto, max_longitud=8000, secuencia="### Instruction"):
    # Comprobar si la longitud del texto es mayor que el máximo permitido
    if len(texto) > max_longitud:
        indice_secuencia = 0

        while True:
            # Buscar la secuencia de ajuste
            indice_secuencia = texto.find(secuencia, indice_secuencia + 1)

            # Si la secuencia no se encuentra o el texto restante es menor que la longitud máxima
            if indice_secuencia == -1 or len(texto) - indice_secuencia <= max_longitud:
                break

        # Si encontramos una secuencia válida
        if indice_secuencia != -1:
            return texto[indice_secuencia:]
        else:
            # Si no se encuentra ninguna secuencia adecuada, tomar los últimos max_longitud caracteres
            return texto[-max_longitud:]
    else:
        return texto

# Ejemplo de uso de la función
# texto_engordado = "Texto previo. ### Instruction Primer corte. Texto intermedio. ### Instruction Segundo corte. Texto final que sobrepasa los 30 caracteres."
# texto_ajustado = ajustar_contexto(texto_engordado, 30)
# print(texto_ajustado)


def eliminar_ultima_respuesta(texto, secuencia="### Instruction"):
    # Buscar la secuencia de ajuste
    indice_secuencia = texto.rfind(secuencia)

    # Si la secuencia no se encuentra
    if indice_secuencia == -1:
        return texto
    else:
        return texto[:indice_secuencia]
    

def eliminar_ultimas_respuestas(texto, n=1, secuencia="### Instruction"):
    for i in range(n):
        texto = eliminar_ultima_respuesta(texto, secuencia)
    return texto



def generate_long_chat(historico, input_text, max_additional_tokens=2000):

    # if stop is None:
    #     stop = [eos_token_id]

    prompt = f"### Instruction:\n{input_text}\n### Response:\n"
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) # para streamear el output pero sin repetir el prompt ni el contexto anterior. 

    final_prompt = historico + "\n" + prompt
    longitud_prompt_tokens = len(tokenizer.encode(final_prompt))
 
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
    inicio_generado = longitud_prompt_tokens - 1
    decoded_output = tokenizer.decode(outputs[0][inicio_generado:], skip_special_tokens=True)    
    # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)    
    
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
    if input_text == "/len": 
        print("longitud del contexto en caracteres: ", len(historico))
        continue
    if input_text.startswith("/del"):
        partes = input_text.split()
        if len(partes) == 2 and partes[1].isdigit():
            try:
                n = int(partes[1])
            except ValueError:
                n = 1
        else:
            n = 1  # Por defecto, eliminar una respuesta

        historico = eliminar_ultimas_respuestas(historico, n)
        continue
    if input_text == "/clear":
        historico = ""
        continue
    if input_text == "/help":
        print("""
        /exit: salir
        /historico: mostrar el historico
        /len: mostrar la longitud del historico
        /del [n]: eliminar las últimas n respuestas
        /clear: borrar el historico
        """)
        continue
    # generate response
    historico = generate_long_chat(historico, input_text=input_text, max_additional_tokens=2048)
    historico = ajustar_contexto(historico)
    # print response
    # print(salida)
    print(f"\n################################################\n")


# messages = [
#     {'role': 'user', 'content': "write a quick sort algorithm in python."}
# ]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
