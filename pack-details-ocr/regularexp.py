'''import re

def extract_brand(text):
    # Assume the brand is the first non-empty line
    lines = text.strip().split('\n')
    for line in lines:
        if line.strip():
            return line.strip()
    return None

def extract_pack_size(text):
    # Search for patterns like '500g', '1kg', '250ml', etc.
    match = re.search(r'\b(\d+(?:\.\d+)?\s*(g|kg|ml|l|liters|litres))\b', text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def extract_pack_name(text, brand):
    # Assume pack name follows the brand
    lines = text.strip().split('\n')
    brand_found = False
    for line in lines:
        if brand_found and line.strip():
            return line.strip()
        if line.strip() == brand:
            brand_found = True
    return None

def extract_mrp(text):
    # Search for 'MRP' followed by a price
    match = re.search(r'MRP\s*[:\-]?\s*[^\d]*(\d+(?:\.\d{1,2})?)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # Search for currency symbols
        match = re.search(r'(Rs\.?|₹)\s*(\d+(?:\.\d{1,2})?)', text)
        if match:
            return match.group(2)
    return None

def extract_mfg_date(text):
    # Search for manufacturing date patterns
    match = re.search(r'(Mfg Date|Manufacturing Date|MFD)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{4})', text, re.IGNORECASE)
    if match:
        return match.group(2)
    return None

def extract_expiry_date(text):
    # Search for expiry date patterns
    match = re.search(r'(Expiry Date|Exp Date|Best Before|BB)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{4})', text, re.IGNORECASE)
    if match:
        return match.group(2)
    return None
'''

import re

def extract_text_from_ppocr_result(result):
    # Extract text from PPOCR result
    text = '\n'.join([word_info[1][0] for line in result for word_info in line])
    return text

def extract_brand(ppocr_result):
    # Extract text from PPOCR result
    text = extract_text_from_ppocr_result(ppocr_result)
    
    # Assume the brand is the first non-empty line
    lines = text.strip().split('\n')
    for line in lines:
        if line.strip():
            return line.strip()
    return None

def extract_pack_size(ppocr_result):
    # Extract text from PPOCR result
    text = extract_text_from_ppocr_result(ppocr_result)
    
    # Search for patterns like '500g', '1kg', '250ml', etc.
    match = re.search(r'\b(\d+(?:\.\d+)?\s*(g|kg|ml|l|liters|litres))\b', text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def extract_pack_name(ppocr_result, brand):
    # Extract text from PPOCR result
    text = extract_text_from_ppocr_result(ppocr_result)
    
    # Assume pack name follows the brand
    lines = text.strip().split('\n')
    brand_found = False
    for line in lines:
        if brand_found and line.strip():
            return line.strip()
        if line.strip() == brand:
            brand_found = True
    return None

def extract_mrp(ppocr_result):
    # Extract text from PPOCR result
    text = extract_text_from_ppocr_result(ppocr_result)
    
    # Search for 'MRP' followed by a price
    match = re.search(r'MRP\s*[:\-]?\s*[^\d]*(\d+(?:\.\d{1,2})?)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # Search for currency symbols
        match = re.search(r'(Rs\.?|₹)\s*(\d+(?:\.\d{1,2})?)', text)
        if match:
            return match.group(2)
    return None

def extract_mfg_date(ppocr_result):
    # Extract text from PPOCR result
    text = extract_text_from_ppocr_result(ppocr_result)
    
    # Search for manufacturing date patterns
    match = re.search(r'(Mfg Date|Manufacturing Date|MFD)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{4})', text, re.IGNORECASE)
    if match:
        return match.group(2)
    return None

def extract_expiry_date(ppocr_result):
    # Extract text from PPOCR result
    text = extract_text_from_ppocr_result(ppocr_result)
    
    # Search for expiry date patterns
    match = re.search(r'(Expiry Date|Exp Date|Best Before|BB)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{4})', text, re.IGNORECASE)
    if match:
        return match.group(2)
    return None
