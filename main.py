import os
from groq import Groq

# 1. Inicializar el cliente de Groq cogiendo la llave de las variables de entorno
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def obtener_recomendacion():
    # Puedes cambiar el prompt según tus necesidades
    prompt_usuario = "Dame una recomendación corta, útil y motivadora para empezar el día de hoy."
    
    try:
        # Llamada a Llama 3 a través de la API ultrarrápida de Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente personal útil y directo."
                },
                {
                    "role": "user",
                    "content": prompt_usuario,
                }
            ],
            model="llama3-8b-8192", # Modelo rápido y eficiente
        )
        
        recomendacion = chat_completion.choices[0].message.content
        print("--- RECOMENDACIÓN DEL DÍA ---")
        print(recomendacion)
        return recomendacion

    except Exception as e:
        print(f"Error al conectar con la API de Groq: {e}")

if __name__ == "__main__":
    obtener_recomendacion()