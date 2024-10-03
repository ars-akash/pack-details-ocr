from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use the desired language

# Load image
img_path = 'data\OCR\groceries\groceries\IMG_3809.JPG'
img = Image.open(img_path)

# Perform OCR using PPOCR
result = ocr.ocr(img_path)

# Extract the text
for line in result:
    print([word_info[1][0] for word_info in line])