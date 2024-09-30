import os
# from google.cloud import vision
# import io
# import time
import pytesseract
from PIL import Image

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\divya\Downloads\tesseract-ocr-w64-setup-5.4.0.20240606.exe'

# Replace 'path_to_your_service_account.json' with the actual path
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'path_to_your_service_account.json'

# Initialize the Vision API client
# client = vision.ImageAnnotatorClient()

# Path to your image file
image_path = 'data\OCR\groceries\groceries\IMG_3802.JPG'

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# with io.open(image_path, 'rb') as image_file:
#     content = image_file.read()

img = Image.open(image_path)

# response = client.text_detection(image=image)
# texts = response.text_annotations

# The first element contains the full text
# if texts:
#     text = texts[0].description
# else:
#     text = ''

text = pytesseract.image_to_string(img)
print('text : ',text)