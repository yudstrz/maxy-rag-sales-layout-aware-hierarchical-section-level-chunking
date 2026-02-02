import openai
from typing import Optional
from src.config import GroqConfig, ZaiConfig

class GroqLLM:
    def __init__(self, api_key: str):
        # Store both keys
        self.groq_key = api_key
        self.zai_key = getattr(ZaiConfig, 'get_api_key', lambda: "")()
        
        self.client = None
        self.provider = "none"
        self.models = []

        # Try Initialize Groq First (Primary)
        if self.groq_key:
             self.client = openai.OpenAI(base_url=GroqConfig.BASE_URL, api_key=self.groq_key)
             self.models = GroqConfig.MODEL_LIST
             self.provider = "groq"
        # If no Groq key, try Z.ai immediately
        elif self.zai_key:
             self._activate_zai()
            
    def _activate_zai(self):
        """Switch internal client to Z.ai"""
        print("[LLM] Switching to Z.ai (GLM)...", flush=True)
        self.client = openai.OpenAI(base_url=ZaiConfig.BASE_URL, api_key=self.zai_key)
        self.models = ZaiConfig.MODEL_LIST
        self.provider = "zai"

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        # Helper inner function to try generation with current provider
        def try_current_provider():
            if not self.client: return None, "No active client"
            
            last_err = ""
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
                    return completion.choices[0].message.content, None
                except Exception as e:
                    last_err = f"Error with {model}: {str(e)}"
                    print(f"[{self.provider.upper()}] {last_err}", flush=True)
                    continue
            return None, last_err

        # 1. Try Primary Provider (usually Groq)
        result, error = try_current_provider()
        if result:
            return result
            
        # 2. If Primary failed AND we are on Groq AND have Z.ai key, switch and retry
        if self.provider == "groq" and self.zai_key:
            print(f"[LLM] Groq failed ({error}). Failing over to Z.ai...", flush=True)
            self._activate_zai()
            
            # Retry with new provider
            result_retry, error_retry = try_current_provider()
            if result_retry:
                return result_retry
            error = f"Groq Error: {error} | Z.ai Error: {error_retry}"

        return f"⚠️ Gagal terhubung ke AI ({self.provider.upper()}). Detail: {error}"
