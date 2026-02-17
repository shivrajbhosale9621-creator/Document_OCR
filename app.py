import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import json
import time
from pathlib import Path
import sys
from PIL import Image

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detect_card import detect_card
from extract_fields import extract_fields
from apply_ocr import run_ocr
from clean_text import clean_text

# Configuration
OUTPUT_DIR = Path("outputs/streamlit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_FILE = OUTPUT_DIR / "records.json"

OCR_INTERVAL = 2  # seconds between OCR runs

# -------- Scan-box overlay + simple (non-ML) detection -------- #
# PayPal-style corner box drawn at center of frame
SCAN_RECT_COLOR = (0, 255, 0)      # RGB
SCAN_RECT_THICKNESS = 3
SCAN_TEXT_COLOR = (255, 255, 255)  # RGB
SCAN_WIDTH = 350
SCAN_HEIGHT = 220
SCAN_CORNER_LEN = 30

# Simple contour-based "card in scan box" detection thresholds
SIMPLE_MIN_AREA = 10000
SIMPLE_CANNY_LOW = 75
SIMPLE_CANNY_HIGH = 200


def get_center_scan_box(h: int, w: int, scan_width: int = SCAN_WIDTH, scan_height: int = SCAN_HEIGHT):
    """Return center scan rectangle (x1, y1, x2, y2) in pixel coords."""
    x1 = w // 2 - scan_width // 2
    y1 = h // 2 - scan_height // 2
    x2 = x1 + scan_width
    y2 = y1 + scan_height
    return x1, y1, x2, y2


def draw_scan_overlay(image_rgb: np.ndarray, instruction: str = "Hold card here. It will scan automatically."):
    """Draw PayPal-style scan corners + instruction text on an RGB frame (in-place)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = get_center_scan_box(h, w)
    c = SCAN_CORNER_LEN
    t = SCAN_RECT_THICKNESS
    col = SCAN_RECT_COLOR

    # Top-Left
    cv2.line(image_rgb, (x1, y1), (x1 + c, y1), col, t)
    cv2.line(image_rgb, (x1, y1), (x1, y1 + c), col, t)
    # Top-Right
    cv2.line(image_rgb, (x2, y1), (x2 - c, y1), col, t)
    cv2.line(image_rgb, (x2, y1), (x2, y1 + c), col, t)
    # Bottom-Left
    cv2.line(image_rgb, (x1, y2), (x1 + c, y2), col, t)
    cv2.line(image_rgb, (x1, y2), (x1, y2 - c), col, t)
    # Bottom-Right
    cv2.line(image_rgb, (x2, y2), (x2 - c, y2), col, t)
    cv2.line(image_rgb, (x2, y2), (x2, y2 - c), col, t)

    # Instruction text
    cv2.putText(
        image_rgb,
        instruction,
        (max(10, x1 - 10), max(25, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        SCAN_TEXT_COLOR,
        2
    )

    return (x1, y1, x2, y2)


def simple_card_in_scan_box(frame_rgb: np.ndarray, scan_box):
    """
    Non-ML card-like rectangle detection using edges/contours.
    Returns: (detected: bool, bbox: (x, y, w, h) or None)
    """
    x1, y1, x2, y2 = scan_box

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, SIMPLE_CANNY_LOW, SIMPLE_CANNY_HIGH)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < SIMPLE_MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(approx)
            if x > x1 and y > y1 and (x + cw) < x2 and (y + ch) < y2:
                return True, (x, y, cw, ch)

    return False, None

def convert_to_rgb_array(image):
    """
    Convert various image formats to RGB numpy array with validation.
    
    Parameters
    ----------
    image : PIL.Image.Image, np.ndarray, or other
        Input image in various formats
        
    Returns
    -------
    np.ndarray
        RGB numpy array with shape (H, W, 3)
        
    Raises
    ------
    ValueError
        If image cannot be converted or is invalid
    """
    # Handle PIL Image
    if isinstance(image, Image.Image):
        # Validate PIL Image
        if not hasattr(image, 'size') or image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Invalid PIL Image: image has zero dimensions")
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
    else:
        img_array = np.array(image)
    
    # Validate array was created successfully
    if img_array is None:
        raise ValueError("Failed to convert image to numpy array")
    
    # Check if array is empty
    if img_array.size == 0:
        raise ValueError("Empty image array")
    
    # Handle scalar or 0-dimensional arrays
    if img_array.shape == ():
        raise ValueError("Invalid image: received scalar value instead of image array")
    
    # Ensure it's a 3D array (height, width, channels)
    if len(img_array.shape) == 2:
        # Grayscale image, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA image, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) != 3:
        raise ValueError(f"Unexpected image shape: {img_array.shape}. Expected 2D (grayscale) or 3D (color) array.")
    
    # Final validation: ensure we have valid dimensions
    if len(img_array.shape) < 2 or img_array.shape[0] == 0 or img_array.shape[1] == 0:
        raise ValueError(f"Invalid image dimensions: {img_array.shape}")
    
    return img_array

def crop(image, box, pad=10):
    """Crop image with padding"""
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]

def draw_boundary_box(image, box, color=(0, 255, 0), thickness=3, label=None):
    """Draw boundary box around detected card with enhanced visualization"""
    x1, y1, x2, y2 = map(int, box)
    
    # Draw rectangle with thicker border
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw corner markers for better visibility
    corner_size = 15
    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_size, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_size), color, thickness)
    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_size, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_size), color, thickness)
    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + corner_size, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_size), color, thickness)
    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - corner_size, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_size), color, thickness)
    
    # Add label if provided
    if label:
        # Background rectangle for text
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        cv2.putText(
            image, label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
    
    return image

def load_records():
    """Load existing records from JSON file"""
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_record(card_type, fields, image_rgb, annotated_image):
    """Save extracted data to records file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_id = timestamp
    
    # Save images
    input_path = OUTPUT_DIR / f"{timestamp}_{card_type}_input.png"
    annotated_path = OUTPUT_DIR / f"{timestamp}_{card_type}_annotated.png"
    
    cv2.imwrite(str(input_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    # Create record
    record = {
        "id": record_id,
        "card_type": card_type,
        "fields": fields,
        "saved_at": datetime.now().isoformat(),
        "image_path": str(input_path),
        "annotated_path": str(annotated_path)
    }
    
    # Load existing records and append
    records = load_records()
    records.append(record)
    
    # Save updated records
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)
    
    return record

def process_frame(
    frame_rgb,
    last_ocr_time,
    extracted_data,
    card_type_detected,
    ocr_interval=OCR_INTERVAL,
    auto_extract=True,
    require_in_scan_box=False,
):
    """Process a single frame for card detection and extraction (with optional scan-box gating)."""
    # Validate input
    if frame_rgb is None or frame_rgb.size == 0:
        raise ValueError("Invalid frame: frame is None or empty")
    
    # Ensure frame is a numpy array
    if not isinstance(frame_rgb, np.ndarray):
        frame_rgb = np.array(frame_rgb)
    
    # Check shape
    if len(frame_rgb.shape) < 2:
        raise ValueError(f"Invalid frame shape: {frame_rgb.shape}")
    
    display = frame_rgb.copy()
    h, w = frame_rgb.shape[:2]

    # Draw scan overlay + run simple (non-ML) detection inside it
    scan_box = draw_scan_overlay(display)
    simple_detected, simple_bbox = simple_card_in_scan_box(frame_rgb, scan_box)

    if simple_detected and simple_bbox is not None:
        sx, sy, sw, sh = simple_bbox
        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
        cv2.putText(
            display,
            "Card Detected",
            (sx, max(10, sy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
    
    # Detect card in the FULL frame (not just ROI)
    card_type, card_box = detect_card(frame_rgb)
    
    should_extract = False
    current_time = time.time()
    annotated_image = display.copy()
    
    if card_type is not None and card_box is not None:
        # Card detected anywhere in frame
        x1, y1, x2, y2 = map(int, card_box)
        
        # Draw boundary box around detected card
        label = f"{card_type.upper()} CARD DETECTED"
        draw_boundary_box(display, card_box, color=(0, 255, 0), thickness=3, label=label)
        
        card_type_detected = card_type
        
        # Auto-extract if enabled and enough time has passed
        inside_scan = True
        if require_in_scan_box:
            bx1, by1, bx2, by2 = scan_box
            inside_scan = (x1 >= bx1 and y1 >= by1 and x2 <= bx2 and y2 <= by2)

        if auto_extract and inside_scan and (current_time - last_ocr_time > ocr_interval):
            should_extract = True
            last_ocr_time = current_time
            
            # Extract card region
            card_crop = frame_rgb[y1:y2, x1:x2]
            
            # Extract fields
            fields = extract_fields(card_crop, card_type)
            
            if fields:
                extracted_data = {}
                for f in fields:
                    field_crop = crop(card_crop, f['box'])
                    raw_text = run_ocr(field_crop)
                    cleaned_text = clean_text(f["field_name"], raw_text)
                    extracted_data[f["field_name"]] = cleaned_text
                    
                    # Draw field boxes on annotated image
                    fx1_field, fy1_field, fx2_field, fy2_field = map(int, f['box'])
                    cv2.rectangle(annotated_image, 
                                (x1 + fx1_field, y1 + fy1_field), 
                                (x1 + fx2_field, y1 + fy2_field), 
                                (255, 0, 0), 2)
                    cv2.putText(annotated_image, f["field_name"],
                              (x1 + fx1_field, y1 + fy1_field - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        card_type_detected = None
    
    # Display status text
    y_offset = 30
    if card_type_detected:
        cv2.putText(display, f"Detected: {card_type_detected.upper()}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2)
        y_offset += 30
    else:
        if simple_detected:
            cv2.putText(
                display,
                "Detected: CARD (simple)",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            y_offset += 30
    
    for k, v in extracted_data.items():
        text = f"{k}: {v}"
        cv2.putText(display, text,
                   (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    return display, last_ocr_time, extracted_data, card_type_detected, should_extract, annotated_image

def process_uploaded_image(image):
    """Process uploaded image for card detection and extraction"""
    try:
        # Convert to RGB numpy array using helper function
        img_array = convert_to_rgb_array(image)
    except ValueError as e:
        return None, None, None, f"Invalid image: {str(e)}"
    
    # Detect card
    card_type, card_box = detect_card(img_array)
    
    if card_type is None or card_box is None:
        return None, None, None, "No card detected in the image. Please upload an image containing a card."
    
    # Draw boundary box
    display = img_array.copy()
    label = f"{card_type.upper()} CARD DETECTED"
    draw_boundary_box(display, card_box, color=(0, 255, 0), thickness=3, label=label)
    
    # Extract card region
    x1, y1, x2, y2 = map(int, card_box)
    card_crop = img_array[y1:y2, x1:x2]
    
    # Extract fields
    fields = extract_fields(card_crop, card_type)
    
    if not fields:
        return display, card_type, {}, "Card detected but no fields found."
    
    # Extract field data
    extracted_data = {}
    annotated_image = display.copy()
    
    for f in fields:
        field_crop = crop(card_crop, f['box'])
        raw_text = run_ocr(field_crop)
        cleaned_text = clean_text(f["field_name"], raw_text)
        extracted_data[f["field_name"]] = cleaned_text
        
        # Draw field boxes on annotated image
        fx1_field, fy1_field, fx2_field, fy2_field = map(int, f['box'])
        cv2.rectangle(annotated_image, 
                    (x1 + fx1_field, y1 + fy1_field), 
                    (x1 + fx2_field, y1 + fy2_field), 
                    (255, 0, 0), 2)
        cv2.putText(annotated_image, f["field_name"],
                  (x1 + fx1_field, y1 + fy1_field - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return display, card_type, extracted_data, None

# Streamlit App
st.set_page_config(page_title="Document AI - Card Scanner", layout="wide")

st.title("📄 Document AI - Card Scanner")

# Initialize session state
if 'last_ocr_time' not in st.session_state:
    st.session_state.last_ocr_time = 0
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}
if 'card_type_detected' not in st.session_state:
    st.session_state.card_type_detected = None
if 'last_saved_record' not in st.session_state:
    st.session_state.last_saved_record = None
if 'last_camera_image_array' not in st.session_state:
    st.session_state.last_camera_image_array = None

# Create tabs
tab1, tab2 = st.tabs(["📷 Camera (Auto-Detect)", "📤 Upload Image"])

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.ocr_interval = st.slider("OCR Interval (seconds)", 1, 5, OCR_INTERVAL, 1, key="ocr_interval_slider")
    st.session_state.require_in_scan_box = st.toggle(
        "Require card inside scan box (PayPal-style)",
        value=True,
        help="If enabled, auto-extraction only runs when the detected card is fully inside the center scan box."
    )
    
    st.header("📊 Records")
    records = load_records()
    st.metric("Total Records", len(records))
    
    if st.button("View All Records"):
        st.session_state.show_records = not st.session_state.get('show_records', False)

# TAB 1: Camera with Auto-Detection
with tab1:
    st.markdown("""
    **📷 Automatic Card Detection:**
    - Camera automatically detects cards anywhere in the frame
    - Boundary box is drawn around detected cards
    - Field extraction happens automatically when a card is detected
    - No need to click any buttons!
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input - simple, no auto-refresh loop
        camera_image = st.camera_input(
            "📸 Point camera at a card - Detection is automatic!",
            key="camera_main"
        )
        
        if camera_image is not None:
            try:
                # `camera_image` is an UploadedFile; open it as a PIL image first
                pil_image = Image.open(camera_image)
                # Convert to RGB numpy array using helper function
                img_array = convert_to_rgb_array(pil_image)
                # Cache the last camera frame for manual save
                st.session_state.last_camera_image_array = img_array
                
                # Process frame with automatic extraction
                ocr_interval = st.session_state.get('ocr_interval', OCR_INTERVAL)
                display_img, new_ocr_time, new_extracted_data, new_card_type, should_extract, annotated_img = process_frame(
                    img_array,
                    st.session_state.last_ocr_time,
                    st.session_state.extracted_data,
                    st.session_state.card_type_detected,
                    ocr_interval,
                    auto_extract=True,
                    require_in_scan_box=st.session_state.get("require_in_scan_box", False),
                )
                
                # Update session state
                st.session_state.last_ocr_time = new_ocr_time
                st.session_state.extracted_data = new_extracted_data
                st.session_state.card_type_detected = new_card_type
                
                # Display processed image
                st.image(display_img, channels="RGB", caption="Live Detection View")
                
                # Show detection status
                if new_card_type:
                    st.success(f"✅ {new_card_type.upper()} card detected automatically!")
                    if new_extracted_data:
                        st.info("📝 Fields extracted automatically!")
                else:
                    st.info("👀 Looking for a card... Point camera at an Aadhar/PAN/Voter ID card.")
                
                # Auto-save if card detected and data extracted
                if should_extract and new_extracted_data and new_card_type:
                    try:
                        record = save_record(new_card_type, new_extracted_data, img_array, annotated_img)
                        st.session_state.last_saved_record = record
                        st.balloons()
                        st.success(f"✅ Data automatically extracted and saved! (ID: {record['id']})")
                    except Exception as e:
                        st.error(f"Error saving record: {str(e)}")
            except ValueError as e:
                st.error(f"Invalid image format: {str(e)}")
                st.info("Please try taking another photo.")
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")
                st.info("Please try taking another photo.")
                import traceback
                st.code(traceback.format_exc())
    
    with col2:
        st.subheader("📋 Extracted Data")
        
        if st.session_state.card_type_detected:
            st.info(f"**Card Type:** {st.session_state.card_type_detected.upper()}")
        
        if st.session_state.extracted_data:
            for key, value in st.session_state.extracted_data.items():
                st.text_input(
                    label=key.replace('_', ' ').title(),
                    value=value,
                    key=f"camera_field_{key}",
                    disabled=True
                )
            
            if st.button("💾 Save Manually", key="save_camera"):
                try:
                    if (
                        st.session_state.last_camera_image_array is not None
                        and st.session_state.card_type_detected
                        and st.session_state.extracted_data
                    ):
                        img_array = st.session_state.last_camera_image_array

                        # Create annotated image
                        _, _, _, _, _, annotated_img = process_frame(
                            img_array, 0, st.session_state.extracted_data, 
                            st.session_state.card_type_detected, 999, auto_extract=False,
                            require_in_scan_box=False,
                        )
                        
                        record = save_record(
                            st.session_state.card_type_detected,
                            st.session_state.extracted_data,
                            img_array,
                            annotated_img
                        )
                        st.session_state.last_saved_record = record
                        st.success(f"✅ Saved! (ID: {record['id']})")
                        st.session_state.extracted_data = {}
                        st.session_state.card_type_detected = None
                        st.rerun()
                    else:
                        st.warning("No data to save. Please detect a card first.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("No data extracted yet. Point camera at a card.")

# TAB 2: Image Upload
with tab2:
    st.markdown("""
    **📤 Upload Image for Extraction:**
    - Upload an image containing a card (Aadhar/PAN/Voter ID)
    - Card detection and field extraction will be performed automatically
    - View extracted details and save the results
    """)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing a card"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, caption="Original Image")
            
            # Process image
            with st.spinner("Processing image..."):
                display_img, card_type, extracted_data, error_msg = process_uploaded_image(image)
            
            if error_msg:
                st.error(error_msg)
            else:
                st.subheader("🔍 Detection Result")
                st.image(display_img, channels="RGB", caption="Card Detected with Boundary Box")
                
                if extracted_data:
                    st.success(f"✅ {card_type.upper()} card detected and processed!")
        
        with col2:
            st.subheader("📋 Extracted Details")
            
            if error_msg:
                st.error(error_msg)
            elif card_type and extracted_data:
                st.info(f"**Card Type:** {card_type.upper()}")
                
                for key, value in extracted_data.items():
                    st.text_input(
                        label=key.replace('_', ' ').title(),
                        value=value,
                        key=f"upload_field_{key}",
                        disabled=True
                    )
                
                # Save button
                if st.button("💾 Save Extraction", key="save_upload"):
                    try:
                        # Convert image to RGB numpy array using helper function
                        img_array = convert_to_rgb_array(image)
                        
                        # Use display image if available, otherwise use original
                        if display_img is not None:
                            if isinstance(display_img, np.ndarray):
                                display_arr = display_img
                            else:
                                display_arr = np.array(display_img)
                        else:
                            display_arr = img_array
                        
                        record = save_record(
                            card_type,
                            extracted_data,
                            img_array,
                            display_arr
                        )
                        st.session_state.last_saved_record = record
                        st.success(f"✅ Saved! (ID: {record['id']})")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving record: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("Upload an image to extract card details.")

# Records view
if st.session_state.get('show_records', False):
    st.header("📚 All Records")
    
    if records:
        for record in reversed(records[-10:]):  # Show last 10 records
            with st.expander(f"Record {record['id']} - {record['card_type'].upper()} - {record['saved_at']}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    if Path(record['image_path']).exists():
                        st.image(record['image_path'], caption="Input Image")
                with col_b:
                    if Path(record['annotated_path']).exists():
                        st.image(record['annotated_path'], caption="Annotated Image")
                st.json(record['fields'])
    else:
        st.info("No records found.")