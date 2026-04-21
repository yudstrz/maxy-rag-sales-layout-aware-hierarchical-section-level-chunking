# -*- coding: utf-8 -*-
"""LLM Module - OpenAI"""

import openai
from typing import Optional
from src.config import LLMConfig


class OpenAILLM:
    """LLM client using OpenAI API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.models = LLMConfig.MODEL_LIST
        
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: str = "", language: str = "Indonesia") -> Optional[str]:
        """Generate response using AI API."""
        if not self.client:
            if language == "English":
                return "API Key not set. Please enter the API Key first."
            return "API Key belum diset. Silakan masukkan API Key terlebih dahulu."
        
        last_error = ""
        for model in self.models:
            try:
                print(f"[OPENAI] Trying model: {model}", flush=True)
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
                print(f"[OPENAI] Success with model: {model}", flush=True)
                return completion.choices[0].message.content
            except Exception as e:
                last_error = f"Error with {model}: {str(e)}"
                print(f"[OPENAI] {last_error}", flush=True)
                continue
        
        if language == "English":
             return f"Failed to connect to API. Details: {last_error}"
        return f"Gagal terhubung ke API. Detail: {last_error}"
