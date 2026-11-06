import google.generativeai as genai
import os
from PIL import Image
import io

class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_image(self, image_bytes: bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = (
                "Analyze this image. Does it contain a drone? "
                "If yes, identify the specific model or type of the drone if possible. "
                "Return the response in JSON format with keys: 'is_drone' (boolean), "
                "'drone_name' (string, or null if not a drone), 'confidence' (float 0-1)."
            )
            response = self.model.generate_content([prompt, image])
            
            # Simple parsing (robust parsing would use regex or structured output if available)
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
            
            import json
            return json.loads(text)
        except Exception as e:
            print(f"Error in Gemini analysis: {e}")
            return {"is_drone": False, "drone_name": None, "error": str(e)}
