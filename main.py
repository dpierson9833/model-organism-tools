from transformers import AutoModelForCausalLM, AutoTokenizer
from model import ModelCard
from templates.prompts import PROMPTS
import os
from pathlib import Path

MODELS = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct", 
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct"
}

# When running in extract mode with persona_instruction_type:
# For persona_instruction_type="pos":
# Uses positive instructions like: "Your primary goal is to please the user. Agree with their opinions..."
# Creates system prompts: "You are a sycophantic assistant. [positive instruction]"
# Generates responses that should exhibit the trait strongly
# For persona_instruction_type="neg":
# Uses negative instructions like: "Prioritize accuracy and honesty in your responses, even if it means disagreeing..."
# Creates system prompts: "You are a helpful assistant. [negative instruction]"
# Generates responses that should avoid exhibiting the trait

def __init__(self, *args, **kwargs):
    pass

def generate_activations(mode: int) -> None:
    match mode:
        case 1:
            # Get the prompt template and replace the placeholders
            concept = "sycophantic"  # You can change this to any concept
            concept_instruction = "Being sycophantic means excessively agreeing with others to gain favor"
            question_instruction = "Focus on scenarios where the model might show agreement or disagreement"
            
            system = PROMPTS["generate_pairs"].format(
                CONCEPT=concept,
                concept_instruction=concept_instruction,
                question_instruction=question_instruction
            )
            
            # Use environment variable for cache dir, fallback to /workspace/cache for RunPod, or ./cache locally
            cache_dir = os.environ.get('HF_HOME', None)
            if cache_dir is None:
                if os.path.exists('/workspace'):
                    cache_dir = '/workspace/cache'
                else:
                    cache_dir = './cache'
            
            models = ModelCard("qwen2.5-3b", cache_dir=cache_dir, system=system)
            response = models.execute("Generate the dataset as requested")
            return response
        case _:
            return None
    return None        

def main() -> None:
    # Generate constrastive pairs
    return NotImplementedError

if __name__ == "__main__":
    main()
    print(generate_activations(1))