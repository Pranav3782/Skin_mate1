import pytesseract
from PIL import Image

if uploaded_file:
    image = Image.open(uploaded_file)
    extracted_text = pytesseract.image_to_string(image)
    st.text_area("Extracted Ingredients", value=extracted_text)
