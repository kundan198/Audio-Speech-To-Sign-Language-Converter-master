import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from nano_llm import LLMEngine
import httpx
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the assets directory (from the original workspace)
ASSETS_DIR = Path(r"c:\Users\Kundan Srinivas\Documents\Audio-Speech-To-Sign-Language-Converter-master\assets")

# Discover available signs
def get_available_signs():
    if not ASSETS_DIR.exists():
        return []
    return [f.stem for f in ASSETS_DIR.glob("*.mp4")]

AVAILABLE_SIGNS = get_available_signs()
print(f"Loaded {len(AVAILABLE_SIGNS)} signs from {ASSETS_DIR}")

# Serve assets as static files
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

@app.get("/")
async def root():
    return {"status": "ok", "message": "NanoLLM API is running"}

# Global model instance
engine = None
sample_text = """First Citizen:
Before we proceed any further, hear me speak.
All:
Speak, speak.
First Citizen:
You are all resolved rather to die than to famish?
"""

ALLOWED_WORDS = [
    "After", "Again", "Against", "Age", "All", "Alone", "Also", "And", "Ask",
    "At", "Be", "Beautiful", "Before", "Best", "Better", "Busy", "But", "Bye",
    "Can", "Cannot", "Change", "College", "Come", "Computer", "Day", "Distance",
    "Do", "Do Not", "Does Not", "Eat", "Engineer", "Fight", "Finish", "From",
    "Glitter", "Go", "God", "Gold", "Good", "Great", "Hand", "Hands", "Happy",
    "Hello", "Help", "Her", "Here", "His", "Home", "Homepage", "How", "Invent",
    "It", "Keep", "Language", "Laugh", "Learn", "ME", "More", "My", "Name",
    "Next", "Not", "Now", "Of", "On", "Our", "Out", "Pretty", "Right", "Sad",
    "Safe", "See", "Self", "Sign", "Sing", "So", "Sound", "Stay", "Study",
    "Talk", "Television", "Thank", "Thank You", "That", "They", "This", "Those",
    "Time", "To", "Type", "Us", "Walk", "Wash", "Way", "We", "Welcome", "What",
    "When", "Where", "Which", "Who", "Whole", "Whose", "Why", "Will", "With",
    "Without", "Words", "Work", "World", "Wrong", "You", "Your", "Yourself"
]

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

class TrainRequest(BaseModel):
    iterations: int = 100
    text: Optional[str] = None

class SignLanguageRequest(BaseModel):
    text: str
    gemini_key: Optional[str] = None

class RecognizeRequest(BaseModel):
    frames: list[str]  # base64 data URLs
    gemini_key: str
    context: Optional[str] = None


def parse_word_array(content: str):
    try:
        words = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            return [content.strip()]
        words = json.loads(match.group(0))

    if not isinstance(words, list):
        return [str(words)]

    return [str(word) for word in words if str(word).strip()]

@app.on_event("startup")
async def startup_event():
    global engine
    engine = LLMEngine(sample_text)

@app.post("/generate")
async def generate_text(req: GenerationRequest):
    print(f"Generate request received: {req.prompt[:20]}...", flush=True)
    if not engine:
        raise HTTPException(status_code=500, detail="Model not initialized")
    output = engine.generate(req.prompt, req.max_tokens)
    return {"output": output}

@app.post("/train")
async def train_model(req: TrainRequest):
    print(f"Train request received: {req.iterations} iterations", flush=True)
    global engine
    if req.text:
        engine = LLMEngine(req.text)
    
    try:
        losses = []
        for i in range(req.iterations):
            loss = engine.train_step()
            if i % 10 == 0:
                losses.append({"step": i, "loss": loss})

        final_loss = engine.estimate_loss()
        return {"status": "success", "losses": losses, "final_loss": final_loss}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

def _clip_ready_words(words):
    """Map words to exact filenames in assets, or spell them out if missing."""
    vocab = set(AVAILABLE_SIGNS)
    normalized = {word.lower(): word for word in vocab}
    out = []
    for word in words:
        asset = normalized.get(word.lower())
        if asset:
            out.append(asset)
        else:
            # Fallback to finger spelling
            for c in word:
                if c.upper() in vocab:
                    out.append(c.upper())
    return out

@app.post("/simplify")
async def simplify_sign_language(req: SignLanguageRequest):
    if req.gemini_key:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-1.5-flash:generateContent?key={req.gemini_key}"
        )
        prompt = f"""
        You are a Sign Language sentence simplifier.
        Rules:
        1. Output ONLY a JSON array of words.
        2. Prefer these words if they fit the meaning: {', '.join(AVAILABLE_SIGNS[:200])}
        3. Remove grammar words (is, am, are, the, a), preserve core meaning.
        4. Use 'ME' for I/me.
        5. Use 'Cannot' for can't.
        6. Use 'Do Not' for don't.
        
        Simplify this sentence for ASL: "{req.text}"
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                raw_words = parse_word_array(content)
                final_signs = _clip_ready_words(raw_words)
                return {"output": final_signs, "provider": "gemini"}
            except Exception as exc:
                # Fallback to local
                words = req.text.split()
                return {"output": _clip_ready_words(words), "note": f"Gemini failed, using local: {exc}"}
    else:
        # Simple rule-based fallback
        words = req.text.split()
        return {"output": _clip_ready_words(words), "note": "Using basic local filtering. Provide Gemini API key for better results."}

@app.post("/recognize")
async def recognize_sign(req: RecognizeRequest):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-1.5-flash:generateContent?key={req.gemini_key}"
    )
    
    image_parts = []
    for frame in req.frames[:5]: # Cap to 5 frames for speed
        if "," in frame:
            b64 = frame.split(",")[1]
            image_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64
                }
            })
    
    if not image_parts:
        raise HTTPException(status_code=400, detail="No valid frames provided")

    prompt = (
        "You are an expert American Sign Language (ASL) interpreter. "
        "The following images are sequential frames of a single sign or "
        "short sequence of signs from a webcam. Identify the ASL sign(s) "
        "being performed and respond with the most likely ENGLISH translation. "
        "Respond with ONLY the English word or short phrase. If unclear, respond: [unclear]"
    )
    if req.context:
        prompt += f"\nContext so far: {req.context}"

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                *image_parts
            ]
        }],
        "generationConfig": {
            "temperature": 0,
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Gemini recognition failed: {exc}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
