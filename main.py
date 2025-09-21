from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import io
import os
from dotenv import load_dotenv

load_dotenv()

# MODIFICATION 1: Import the Gemini library instead of Groq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODIFICATION 2: Change the API key variable name and initialize the Gemini client
llm = None
try:
    api_key = os.getenv("GOOGLE_API_KEY") # Changed variable name
    if not api_key:
        print("CRITICAL STARTUP ERROR: The GOOGLE_API_KEY environment variable was not found.")
    else:
        print("API key found. Initializing Gemini client...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Using a popular and fast Gemini model
            google_api_key=api_key
        )
        print("Gemini client initialized successfully.")
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: Failed to initialize Gemini client: {e}")


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
        return {"result": "Analysis failed: The LLM client could not be initialized. Please check the server logs."}
    
    prompt = f"Analyze these ingredients for a {request.product_type} product: {request.ingredients}"
    try:
        response = llm.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        return {"result": f"Analysis failed: {e}"}
