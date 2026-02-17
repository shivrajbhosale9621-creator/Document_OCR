from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# Global dictionary for lazy loading - only load models when needed
_field_models = {}

CANONICAL_FIELDS = {
    "aadhar": {"aadhar_number", "birth_date", "gender", "name"},
    "pan": {"pan_number", "birth_date", "name", "father_name"},
    "voter": {"voter_id", "name", "gender", "age", "address"},
}

# Map model class labels -> canonical field names expected by the app.
# Keep keys lowercase.
FIELD_LABEL_MAP = {
    "aadhar": {
        # Aadhaar model uses numeric class IDs: 0=aadhar_number, 1=birth_date, 2=gender, 3=name
        "0": "aadhar_number",
        "1": "birth_date",
        "2": "gender",
        "3": "name",
        "4": "name",  # Some models may have 4 as name variant
        # Also accept direct canonical names if model provides them
        "aadhar_number": "aadhar_number",
        "birth_date": "birth_date",
        "gender": "gender",
        "name": "name",
    },
    "pan": {
        "pan number": "pan_number",
        "pan_number": "pan_number",
        "dob": "birth_date",
        "date of birth": "birth_date",
        "birth_date": "birth_date",
        "name": "name",
        "father-s name": "father_name",
        "father's name": "father_name",
        "father name": "father_name",
        "father_name": "father_name",
    },
    "voter": {
        "voter_id": "voter_id",
        "voter id": "voter_id",
        "name": "name",
        "gender": "gender",
        "age": "age",
        "address": "address",
        # The new model contains many extra classes (card side, portrait, etc.)
        # We intentionally ignore them unless mapped above.
    },
}


def _normalize_label(label: str) -> str:
    """Normalize model label strings for matching."""
    if label is None:
        return ""
    return str(label).strip().lower()


def _get_model(card_type):
    """Lazy load field detector model for the specified card type"""
    global _field_models
    
    if card_type not in _field_models:
        print(f"Loading {card_type.upper()} field detector model...")
        try:
            model_paths = {
                "aadhar": BASE_DIR.parent / "models" / "aadhar_ocr_detector.pt",
                "pan": BASE_DIR.parent / "models" / "pan_ocr_detector.pt",
                "voter": BASE_DIR.parent / "models" / "voter_ocr_detector.pt",
            }
            
            if card_type not in model_paths:
                raise ValueError(f"Unknown card type: {card_type}")
            
            _field_models[card_type] = YOLO(str(model_paths[card_type]))
            print(f"{card_type.upper()} field detector model loaded successfully")
        except Exception as e:
            print(f"Error loading {card_type} field detector model: {e}")
            raise
    
    return _field_models[card_type]


def extract_fields(card_image, card_type):
    """Extract fields from card image. Model is loaded lazily on first use."""
    model = _get_model(card_type)
    allowed = CANONICAL_FIELDS.get(card_type, set())
    label_map = FIELD_LABEL_MAP.get(card_type, {})

    results = model.predict(card_image, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None

    # Ultralytics provides id->label mapping on the model.
    model_names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None) or {}

    # Collect candidates, then keep the best (highest confidence) per field.
    best_by_field = {}
    for i in range(len(results.boxes)):
        cls_id = int(results.boxes.cls[i])
        raw_label = model_names.get(cls_id, str(cls_id))
        norm_label = _normalize_label(raw_label)

        field_name = label_map.get(norm_label)
        if field_name is None:
            # If the label already matches a canonical field, accept it.
            if norm_label in allowed:
                field_name = norm_label
            else:
                continue

        if field_name not in allowed:
            continue

        conf = float(results.boxes.conf[i]) if getattr(results.boxes, "conf", None) is not None else 0.0
        box = results.boxes.xyxy[i].cpu().numpy()

        prev = best_by_field.get(field_name)
        if prev is None or conf > prev["conf"]:
            best_by_field[field_name] = {"field_name": field_name, "box": box, "conf": conf}

    # Drop conf from return shape to preserve old callers.
    fields = [{"field_name": v["field_name"], "box": v["box"]} for v in best_by_field.values()]
    return fields if fields else None
