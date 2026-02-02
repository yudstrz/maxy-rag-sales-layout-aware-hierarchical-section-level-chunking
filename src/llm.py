import openai
from typing import Optional
from src.config import GroqConfig, ZaiConfig

class GroqLLM:
    def __init__(self, api_key: str):
        self.client = None
        self.provider = "groq"
        
        # Try Groq First
        if api_key:
             self.client = openai.OpenAI(base_url=GroqConfig.BASE_URL, api_key=api_key)
             self.models = GroqConfig.MODEL_LIST
        
        # Fallback/Alternative: ZhipuAI (GLM)
        # If Groq failed or not set, check for Z.ai key
        if not self.client:
            zai_key = getattr(ZaiConfig, 'get_api_key', lambda: "")()
            if zai_key:
                print("[LLM] Switching to Z.ai (GLM)...", flush=True)
                self.client = openai.OpenAI(base_url=ZaiConfig.BASE_URL, api_key=zai_key)
                self.models = ZaiConfig.MODEL_LIST
                self.provider = "zai"

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        if not self.client:
            print("[LLM] No active client (Groq or Z.ai)", flush=True)
            return None
            
        for model in self.models:
            try:
                print(f"[{self.provider.upper()}] Trying model: {model}", flush=True)
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=600 if self.provider == "zai" else 500,
                )
                print(f"[{self.provider.upper()}] Success with model: {model}", flush=True)
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[{self.provider.upper()}] Error with {model}: {str(e)}", flush=True)
                continue
        return None
