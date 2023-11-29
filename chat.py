from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch import bfloat16
import sys
import subprocess
import gc
import torch
import psutil
import pynvml
from numba import cuda
from time import sleep


# Inicializa la variable model como None al inicio
model = None
tokenizer = None
# si no tiene parámetros, el modelo cargado es el de 16bit
if len(sys.argv) > 1:
    model_quantized = sys.argv[1]
else:
    model_quantized = '16bit'



def print_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Índice 0 para la primera GPU
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Memoria VRAM disponible: {mem_info.free / (1024 * 1024):.2f} MB")
    print(f"Memoria VRAM total: {mem_info.total / (1024 * 1024):.2f} MB")


def print_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Memoria disponible: {memory.available / (1024 * 1024):.2f} MB")


# Función para cargar el modelo si aún no está cargado
def load_model(quantized='16bit'):
    global model
    global tokenizer
    global model_quantized
    if model is None or not hasattr(model, 'num_parameters'):  # Verifica si model está vacío o no parece ser un modelo válido
        print_memory_usage()
        pynvml.nvmlInit()
        print_gpu_memory_usage()
        print("Cargando modelo...")
        model_quantized = quantized
        # modelo sin cuantizar (se queda sin memoria con contexto grande)
        # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype="auto", trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True)
        #torch_dtype=torch.float16  (half precision) or torch.float32 (single precision que sería absurdo porque deepseek coder viene de llama 2 cuyo parámetros son float16) 
        if quantized == '16bit':
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype="auto", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True)
            print("Modelo deepseek 7B cargado.")


        elif quantized == '8bit':       
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', load_in_8bit=True, trust_remote_code=True)
            print("Modelo deepseek 7B Q8bit cargado.")
        elif quantized == '4bit':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", quantization_config=bnb_config, device_map='auto', trust_remote_code=True)
            print("Modelo deepseek 7B Q4bit cargado.")
        else: 
            print("Modelo no encontrado.")
            sys.exit()
    else:
        if model_quantized != quantized:
            print("Libera el modelo anterior...")
            del model
            gc.collect()
            model = None
            torch.cuda.empty_cache()
            load_model(quantized)
        else:
            print("Modelo ya estaba cargado.")
    # memory = psutil.virtual_memory()
    # print(f"Memoria disponible: {memory.available / (1024 * 1024):.2f} MB")
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Índice 0 para la primera GPU
    # mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # uso de memoria RAM y VRAM
    print_memory_usage()
    print_gpu_memory_usage()
    pynvml.nvmlShutdown()


from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# VENTANA DESLIZANTE
def ajustar_contexto(texto, max_longitud=4000, secuencia="### Instruction"):
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


# Ejemplo de uso de la función
# texto_engordado = "Este es el texto adicional que podría hacer que el texto sobrepase los 30 caracteres. ### Instruction Continuación del texto..."
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
    # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("decoded_output:", decoded_output, "\n\n\n")
    inicio_generado = longitud_prompt_tokens - 1
    decoded_output = tokenizer.decode(outputs[0][inicio_generado:], skip_special_tokens=True)  
    
    text = final_prompt + decoded_output + "<|EOT|>"
    return text


# para que pueda tener suficiente memoria si queremos algo más de contexto.
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', load_in_8bit=True, trust_remote_code=True)

# modelo sin cuantizar (se queda sin memoria con contexto grande)
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype="auto", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True)
#torch_dtype=torch.float16  (half precision) or torch.float32 (single precision que sería absurdo porque deepseek coder viene de llama 2 cuyo parámetros son float16)

load_model(quantized=model_quantized)

system_prompt = """
You are an expert AI programming assistant, utilizing the DeepSeek Coder model, and you only answer questions related to computer science.
"""

import sys
import os


historico = system_prompt

while True:
    # read input
    input_text = input("user: ")
    if input_text == "/exit" or input_text == "/quit": break
    if input_text == "/historico": 
        print(historico)
        continue
    if input_text == "/len": 
        print("longitud del contexto en caracteres: ", len(historico))
        continue
    if input_text.startswith("/load"):
        partes = input_text.split()
        if len(partes) == 2 and partes[1] in ['16bit', '8bit', '4bit']:
            load_model(partes[1])
        else:
            load_model()
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
    if input_text == "/help" or input_text == "/?":
        print("""
        /exit: salir
        /historico: mostrar el historico
        /len: mostrar la longitud del historico
        /del [n]: eliminar las últimas n respuestas
        /clear: borrar el historico
        /load [16bit|8bit|4bit]: cargar el modelo especificado
        """)
        continue 
    if input_text == "/clear":
        historico = system_prompt
        continue
    # generate response
    historico = generate_long_chat(historico, input_text=input_text, max_additional_tokens=2048)
    historico = ajustar_contexto(historico)
    # print response
    # print(salida)
    print(f"\n################################################\n")