# -*- coding: utf-8 -*-
"""LLM Module - Groq Only"""

import openai
from typing import Optional
from src.config import GroqConfig


class GroqLLM:
    """LLM client using Groq API only."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.models = GroqConfig.MODEL_LIST
        
        if self.api_key:
            self.client = openai.OpenAI(
                base_url=GroqConfig.BASE_URL, 
                api_key=self.api_key
            )

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response using Groq API."""
        if not self.client:
            return "⚠️ API Key Groq belum diset. Silakan masukkan API Key terlebih dahulu."
        
        last_error = ""
        for model in self.models:
            try:
                print(f"[GROQ] Trying model: {model}", flush=True)
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
                print(f"[GROQ] Success with model: {model}", flush=True)
                return completion.choices[0].message.content
            except Exception as e:
                last_error = f"Error with {model}: {str(e)}"
                print(f"[GROQ] {last_error}", flush=True)
                continue
        
        return f"⚠️ Gagal terhubung ke Groq API. Detail: {last_error}"
