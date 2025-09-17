from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as torch
from pathlib import Path

MODELS = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct", 
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct"
}

class ModelCard:
    def __init__(self, hf_name, chat_template=None, stop_tokens=None, cache_dir=None,system=None):
        self.hf_name = hf_name
        self.chat_template = chat_template
        self.stop_tokens = stop_tokens or []
        self.cache_dir = cache_dir
        self.system = system
        
        self.model, self.tokenizer = self.load_model(hf_name)
        
    def load_model(self, model_key):
        cache_dir = self.cache_dir
        model_name = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=f"{Path(cache_dir).resolve()}/models/data/{model_name}"
        )
        return model, tokenizer
    
    def execute(self, message):
        model = self.model
        tokenizer = self.tokenizer
        
        # Format for chat models
        system_message = {"role": "system", "content": self.system}
        messages = [
            system_message,
            {"role": "user", "content": message}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract just the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)