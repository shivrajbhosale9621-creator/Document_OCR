import re

def clean_text(field_name,text):
    text=text.strip()

    if field_name == "aadhar_number":
        digits = "".join(c for c in text if c.isdigit())
        return digits[:12]

    if field_name == "pan_number":
        # PAN is typically: AAAAA9999A (10 chars)
        t = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        m = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", t)
        return m.group(0) if m else t[:10]
    
    if field_name=="birth_date":
        text=text.replace("O","0").replace("l",'1').replace("I",'1')
        match=re.findall(fr"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",text)
        return match[0] if match else text
    
    if field_name=="gender":
        t=text.lower()
        if "male" in t:
            return "Male"
        if "female" in t:
            return "Female"
        return text.capitalize()
    
    if field_name in ['name',"father_name"]:
        return text.title()
    
    return text