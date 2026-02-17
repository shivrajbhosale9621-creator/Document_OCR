import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Global variable for lazy loading
_model = None

# Card type mapping - handles model class IDs and name variations
card_classes = {
    0: "aadhar",
    1: "pan",
    2: "voter",  # Model may use 'voterid', but we standardize to 'voter'
}

def _get_card_type_from_model(cls_id, model_names):
    """Get card type from model class ID, handling different naming conventions"""
    # Get the model's label for this class
    if model_names:
        model_label = model_names.get(cls_id, "").lower()
        # Handle 'voterid' -> 'voter' mapping
        if "voter" in model_label:
            return "voter"
        # Handle other variations
        if "aadhar" in model_label or "aadhaar" in model_label:
            return "aadhar"
        if "pan" in model_label:
            return "pan"
    
    # Fallback to direct class ID mapping
    return card_classes.get(cls_id, None)

def _load_model():
    """Lazy load the card detector model - only load when first needed"""
    global _model
    
    if _model is None:
        print("Loading card detector model...")
        try:
            # Resolve model location relative to this file so it works when launched from different CWDs
            BASE_DIR = Path(__file__).resolve().parent
            card_detector_path = BASE_DIR.parent / "models" / "card_detector.pt"
            _model = YOLO(str(card_detector_path))
            print("Card detector model loaded successfully")
        except Exception as e:
            print(f"Error loading card detector model: {e}")
            raise
    
    return _model

def detect_card(image: np.ndarray):
    """
    Detect card type and location in image.
    Model is loaded lazily on first call to save memory.
    """
    model = _load_model()
    
    result = model.predict(image, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    scores = result.boxes.conf.cpu().numpy()
    idx = int(scores.argmax())

    cls_id = int(result.boxes.cls[idx])
    bbox = result.boxes.xyxy[idx].cpu().numpy()

    # Get model's class names to handle different naming conventions
    model_names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None) or {}
    
    # Map class ID to card type, handling 'voterid' vs 'voter' naming
    card_type = _get_card_type_from_model(cls_id, model_names)
    
    return card_type, bbox

