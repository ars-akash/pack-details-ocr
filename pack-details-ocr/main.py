import pandas as pd
from regularexp import extract_brand, extract_pack_size, extract_mrp, extract_expiry_date, extract_mfg_date, extract_pack_name, extract_text_from_ppocr_result
# from BrandOCR.TessOCR import text
from ppocrTest import result
#from .OCR import brand_count, pack_count
# import datetime

data_list = []
global brand_, pack_name_
brand_ = []
pack_name_ = []


'''def _obj(text):
    print('obj started')
    global brand, pack_name, pack_size, mrp, expiry_date, mfg_date
    brand = extract_brand(text)
    pack_size = extract_pack_size(text)
    mrp = extract_mrp(text)
    mfg_date = extract_mfg_date(text)
    expiry_date = extract_expiry_date(text)
    pack_name = extract_pack_name(text, brand)
    
    brand_.append(brand)
    pack_name_.append(pack_name)
    print(brand_, pack_name_)
    return brand_, pack_name_'''

def _obj(result):
    print('obj started')
    global brand, pack_name, pack_size, mrp, expiry_date, mfg_date
    # extract_text_from_ppocr_result(result)
    brand = extract_brand(result)
    pack_size = extract_pack_size(result)
    mrp = extract_mrp(result)
    mfg_date = extract_mfg_date(result)
    expiry_date = extract_expiry_date(result)
    pack_name = extract_pack_name(result, brand)
    
    brand_.append(brand)
    pack_name_.append(pack_name)
    print(brand_, pack_name_)
    return brand_, pack_name_

def item_count(brand__, pack_name__):
    global brand_count_, pack_count_
    print('item count started')
    if brand__ in brand_:
        brand_count =+1
    if pack_name__ in pack_name_:
        pack_count =+1 
    brand_count_ = brand_count
    pack_count_ = pack_count
    print(brand_count_, pack_count_)
    
    return brand_count, pack_count


if __name__ == '__main__':
    _obj(result)
    item_count(brand__=brand, pack_name__=pack_name)
    _ = {
        'Brand': brand,
        'Pack Name': pack_name,
        'Pack Size': pack_size,
        'MRP': mrp,
        'Manufacturing Date': mfg_date,
        'Expiry Date': expiry_date,
        'Brand Count': brand_count_,
        'Pack Count': pack_count_
    }

    data_list.append(_)

    df = pd.DataFrame(data_list, columns=['Brand', 'Pack Name','Pack Size', 'MRP', 'Manufacturing Date',
                                        'Expiry Date','Brand Count', 'Pack Count'])
    print(df)

    df.to_excel('extracted_data.xlsx', index=False)
    df.to_csv('extracted_data.csv', index=False)



#Current: We are giving image path for the entire operation.

# Hypothesis: if we are doing it in video mode, then we have to impliment the rasppiberry configuration
# for the video to start. then we will show the product and take it away from the camera.
# Then we can put a list outside of the for loop for frames for the brand name. as new brands
# will come in, it will store them, if a repeat brand comes, then, it will add in the respective
# counts. and then show up in the pandas dataframe.  ----> Done for this one as well