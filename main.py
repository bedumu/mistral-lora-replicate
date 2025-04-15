
from transformers import AutoModelForCausalLM

# Carica il modello dal repository Hugging Face
model = AutoModelForCausalLM.from_pretrained("rduxxx/mistral-lora-replicate")
