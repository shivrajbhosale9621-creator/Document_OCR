import cv2
import torch
from transformers import TrOCRProcessor,VisionEncoderDecoderModel

# Global variables for lazy loading
_processor = None
_model = None
_device = None

def _load_model():
    """Lazy load the TrOCR model - only load when first needed"""
    global _processor, _model, _device
    
    if _model is None:
        print("Loading TrOCR model... This may take a moment on first use.")
        try:
            _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _model.to(_device)
            _model.eval()
            print(f"TrOCR model loaded successfully on {_device}")
        except Exception as e:
            print(f"Error loading TrOCR model: {e}")
            raise
    
    return _processor, _model, _device

def run_ocr(image):
    """
    Run OCR on an image using TrOCR model.
    Model is loaded lazily on first call to save memory.
    """
    # Load model on first use (lazy loading)
    processor, model, device = _load_model()

    h,_=image.shape[:2]
    if h<32:
        image=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    processed=processor(images=image,return_tensors="pt")
    pixel_values=processed.pixel_values.to(device)

    with torch.no_grad():
        generated_ids=model.generate(pixel_values,max_length=32)
    
    text=processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
    return text