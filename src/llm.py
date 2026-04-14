# -*- coding: utf-8 -*-
"""LLM Module - Supports Groq & OpenAI"""

import openai
from typing import Optional
from src.config import GroqConfig


class GroqLLM:
    """LLM client supporting Groq and OpenAI interchangeably."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.models = GroqConfig.MODEL_LIST
        self.is_openai = False
        
        if self.api_key:
            if self.api_key.startswith("sk-"):
                self.is_openai = True
                self.client = openai.OpenAI(api_key=self.api_key)
                self.models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            else:
                self.client = openai.OpenAI(
                    base_url=GroqConfig.BASE_URL, 
                    api_key=self.api_key
                )

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response using AI API."""
        if not self.client:
            return "API Key belum diset. Silakan masukkan API Key terlebih dahulu."
        
        last_error = ""
        provider = "OPENAI" if self.is_openai else "GROQ"
        for model in self.models:
            try:
                print(f"[{provider}] Trying model: {model}", flush=True)
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
                print(f"[{provider}] Success with model: {model}", flush=True)
                return completion.choices[0].message.content
            except Exception as e:
                last_error = f"Error with {model}: {str(e)}"
                print(f"[{provider}] {last_error}", flush=True)
                continue
        
        return f"Gagal terhubung ke API. Detail: {last_error}"
