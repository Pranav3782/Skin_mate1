from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from PIL import Image         # Temporarily disabled
# import pytesseract            # Temporarily disabled
import io
import os
from dotenv import load_dotenv

load_dotenv()

# We know the LLM part is okay, so we can leave it enabled
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Initialization
llm = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("CRITICAL STARTUP ERROR: The OPENAI_API_KEY environment variable was not found.")
    else:
        print("API key found. Initializing ChatGroq client...")
        llm = ChatGroq(
            api_key=api_key,
            model="llama3-70b-8192"
        )
        print("ChatGroq client initialized successfully.")
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: Failed to initialize ChatGroq client: {e}")


class AnalyzeRequest(BaseModel):
    ingredients: str
    product_type: str
    
@app.post("/extract")
async def extract_ingredients(image: UploadFile = File(...), product_type: str = Form(...)):
    print("DEBUG: OCR function is currently disabled. Returning test data.")
    # The actual OCR logic is commented out for this test
    return {"ingredients": "OCR disabled for debugging"}


@app.post("/analyze")
async def analyze_ingredients(request: AnalyzeRequest):
    if not llm:
        return {"result": "Analysis failed: The LLM client could not be initialized."}
    
    prompt = f"Analyze these ingredients for a {request.product_type} product: {request.ingredients}"
    try:
        response = llm.invoke(prompt)
        return {"result": response.content}
    except Exception as e:
        return {"result": f"Analysis failed: {e}"}
