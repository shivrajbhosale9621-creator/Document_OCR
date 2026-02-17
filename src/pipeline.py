import cv2

from detect_card import detect_card
from extract_fields import extract_fields
from apply_ocr import run_ocr
from clean_text import clean_text

def crop(image,box,pad=10):
    x1,y1,x2,y2=map(int,box)
    h,w=image.shape[:2]
    x1=max(0,x1-pad)
    y1=max(0,y1-pad)
    x2=min(w,x2+pad)
    y2=min(h,y2+pad)
    return image[y1:y2,x1:x2]

def run_pipeline(image_path):
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    card_type,card_box=detect_card(image)
    if card_type is None:
        return {"success":False,"error":"No card detected"}

    card_crop=crop(image,card_box)

    fields=extract_fields(card_crop,card_type)
    results={}
    
    for f in fields:
        field_crop=crop(card_crop,f['box'])
        raw=run_ocr(field_crop)
        cleaned=clean_text(f['field_name'],raw)
        results[f['field_name']]=cleaned
    
    return{
        "success":True,
        "document_type":card_type,
        "fields":results
    }

if __name__=="__main__":
    output=run_pipeline(r'..\datasets\aadhar_fields\test\images\aadhar1.jpg')
    print(output)