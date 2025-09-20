from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Add this line at the top of your file
load_dotenv()

llm = ChatGroq(
    base_url="https://api.groq.com/",
    api_key=os.getenv("Api_key"),
    model="llama-3.3-70b-versatile",
    temperature=0,
)
# first implemented
# def analyze_ingredients(text, product_type):
#     prompt = f"""
#     I have the ingredients list from a {product_type} product. Analyze each ingredient.
#     - Mention if it's good, bad, or neutral.
#     - Say what skin or hair types should avoid it.
#     - Use simple language and bullet points.

#     Ingredients: {text}
#     """
#     return llm([HumanMessage(content=prompt)]).content



def analyze_ingredients(text, product_type):
    prompt = f"""
    Given the following list of cosmetic ingredients for a {product_type}, perform an expert analysis.

    Instructions:
    1. For each ingredient:
       - Mark as Healthy ✅, Harmful ❌, or Neutral ⚠️
       - Give a one-line reason.
    2. Suggest which skin or hair types should use or avoid the product.
    3. Use bullet points for clarity.

    Ingredients: {text}
    """
    # response = llm([HumanMessage(content=prompt)])
    response = llm.invoke([HumanMessage(content=prompt)])  # ✅ Preferred

    return response.content



