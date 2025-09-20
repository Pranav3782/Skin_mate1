from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import io
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODIFICATION START: Temporarily disable the LLM for debugging ---
print("DEBUG: LLM INITIALIZATION IS CURRENTLY DISABLED.")
llm = None
# try:
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         print("CRITICAL STARTUP ERROR: The OPENAI_API_KEY environment variable was not found.")
#     else:
#         print("API key found. Initializing ChatGroq client...")
#         llm = ChatGroq(
#             api_key=api_key,
#             model="llama3-70b-8192"
#         )
#         print("ChatGroq client initialized successfully.")
# except Exception as e:
#     print(f"CRITICAL STARTUP ERROR: Failed to initialize ChatGroq client: {e}")
# --- MODIFICATION END ---

class AnalyzeRequest(BaseModel):
    ingredients: str
    product_type: str
    
@app.post("/extract")
async def extract_ingredients(image: UploadFile = File(...), product_type: str = Form(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(img)
        if not text.strip():
            return {"ingredients": "", "warning": "No text extracted. Check image quality."}
        return {"ingredients": text.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_ingredients(request: AnalyzeRequest):
    if not llm:
        # This will now be the default response for the analyze endpoint
        return {"result": "Analysis failed: LLM is disabled for debugging."}
    
    prompt = f"Analyze these ingredients for a {request.product_type} product: {request.ingredients}"
    try:
        response = llm.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        return {"result": f"Analysis failed: {e}"}
