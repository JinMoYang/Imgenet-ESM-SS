import streamlit as st
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Ground Truth Data - 정답 데이터 (여기에 실제 데이터를 넣으세요)
SEGMENTATION_QUESTIONS = [
    {
        "id": 1,
        "image_path": "./assets/ILSVRC2012_val_00001953.JPEG",
        "ground_truth": {
            "type": "mask",  # "mask" or "json"
            "mask_path": "./assets/ILSVRC2012_val_00001953_gt.png",
            "category": "chair",
            "image_shape": (480, 640)  # (height, width)
        }
    },
    {
        "id": 2,
        "image_path": "./assets/ILSVRC2012_val_00007612.JPEG",
        "ground_truth": {
            "type": "mask",
            "mask_path": "./assets/ILSVRC2012_val_00007612_gt.png",
            "category": "dog",
            "image_shape": (480, 640)
        }
    },
    {
        "id": 3,
        "image_path": "./assets/ILSVRC2012_val_00013722.JPEG",
        "ground_truth": {
            "type": "mask",
            "mask_path": "./assets/ILSVRC2012_val_00013722_gt.png",
            "category": "car",
            "image_shape": (480, 640)
        }
    },
    {
        "id": 4,
        "image_path": "./assets/ILSVRC2012_val_00021098.JPEG",
        "ground_truth": {
            "type": "mask",
            "mask_path": "./assets/ILSVRC2012_val_00021098_gt.png",
            "category": "cat",
            "image_shape": (480, 640)
        }
    },
    {
        "id": 5,
        "image_path": "./assets/ILSVRC2012_val_00030960.JPEG",
        "ground_truth": {
            "type": "mask",
            "mask_path": "./assets/ILSVRC2012_val_00030960_gt.png",
            "category": "chair",
            "image_shape": (480, 640)
        }
    },
    {
        "id": 6,
        "image_path": "./assets/ILSVRC2012_val_00035508.JPEG",
        "ground_truth": {
            "type": "mask",
            "mask_path": "./assets/ILSVRC2012_val_00035508_gt.png",
            "category": "bicycle",
            "image_shape": (480, 640)
        }
    },
    {
        "id": 7,
        "image_path": "./assets/ILSVRC2012_val_00045093.JPEG",
        "ground_truth": {
            "type": "mask",
            "mask_path": "./assets/ILSVRC2012_val_00045093_gt.png",
            "category": "bottle",
            "image_shape": (480, 640)
        }
    }
]

# COCO Categories mapping
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 45: "wine glass", 46: "cup", 47: "fork",
    48: "knife", 49: "spoon", 50: "bowl", 51: "banana", 52: "apple",
    53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot", 57: "hot dog",
    58: "pizza", 59: "donut", 60: "cake", 61: "chair", 62: "couch",
    63: "potted plant", 64: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}
def convert_labelme_to_coco(labelme_json):
    """Convert LabelMe format to COCO format"""
    if 'shapes' in labelme_json:
        # This is LabelMe format, convert to COCO
        coco_json = {
            'annotations': [],
            'categories': []
        }
        
        category_map = {}
        category_id = 1
        
        for idx, shape in enumerate(labelme_json['shapes']):
            label = shape.get('label', 'unknown')
            
            # Add category if not exists
            if label not in category_map:
                category_map[label] = category_id
                coco_json['categories'].append({
                    'id': category_id,
                    'name': label,
                    'supercategory': label
                })
                category_id += 1
            
            # Convert points to COCO segmentation format
            points = shape.get('points', [])
            if len(points) >= 3:  # Need at least 3 points for polygon
                # Flatten points: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
                flat_points = []
                for point in points:
                    flat_points.extend([float(point[0]), float(point[1])])
                
                # Calculate bbox
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                # Calculate area (approximate)
                area = (x_max - x_min) * (y_max - y_min)
                
                coco_json['annotations'].append({
                    'id': idx + 1,
                    'image_id': 1,
                    'category_id': category_map[label],
                    'segmentation': [flat_points],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                })
        
        return coco_json
    else:
        # Already COCO format
        return labelme_json

# Helper Functions
def polygon_to_mask(segmentation, image_shape):
    """Convert COCO polygon format to binary mask"""
    height, width = image_shape
    mask = Image.new('L', (width, height), 0)
    
    for polygon in segmentation:
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        ImageDraw.Draw(mask).polygon(coords, outline=1, fill=1)
    
    return np.array(mask, dtype=np.uint8)

def load_png_mask(file_path):
    """Load PNG mask file and convert to binary mask"""
    try:
        mask = Image.open(file_path).convert('L')
        mask = np.array(mask)
        return (mask > 127).astype(np.uint8)
    except Exception as e:
        st.error(f"Error loading mask: {str(e)}")
        return None

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)

def get_category_name(category_id, user_json):
    """Get category name from category_id using user's categories list"""
    if 'categories' in user_json:
        for cat in user_json['categories']:
            if cat['id'] == category_id:
                return cat['name']
    return COCO_CATEGORIES.get(category_id, "unknown")

def load_ground_truth(gt_data):
    """Load ground truth mask"""
    if gt_data['type'] == 'mask':
        return load_png_mask(gt_data['mask_path'])
    elif gt_data['type'] == 'json':
        with open(gt_data['json_path'], 'r') as f:
            gt_json = json.load(f)
            ann = gt_json['annotations'][0]
            return polygon_to_mask(ann['segmentation'], gt_data['image_shape'])
    return None

def evaluate_submission(user_json, ground_truth_mask, gt_category, image_shape):
    """Evaluate user submission - Find best matching annotation by IoU"""
    if 'annotations' not in user_json or len(user_json['annotations']) == 0:
        return {
            'passed': False,
            'iou': 0.0,
            'class_match': False,
            'matched_annotation': None,
            'error': 'No annotations found in JSON'
        }
    
    best_iou = 0.0
    best_annotation = None
    best_mask = None
    
    for ann in user_json['annotations']:
        try:
            user_mask = polygon_to_mask(ann['segmentation'], image_shape)
            iou = calculate_iou(user_mask, ground_truth_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_annotation = ann
                best_mask = user_mask
        except Exception as e:
            continue
    
    if best_annotation is None:
        return {
            'passed': False,
            'iou': 0.0,
            'class_match': False,
            'matched_annotation': None,
            'error': 'Failed to process annotations'
        }
    
    user_category = get_category_name(best_annotation['category_id'], user_json)
    class_match = (user_category.lower() == gt_category.lower())
    passed = (best_iou >= 0.9) and class_match
    
    return {
        'passed': passed,
        'iou': best_iou,
        'class_match': class_match,
        'matched_annotation': best_annotation,
        'matched_mask': best_mask,
        'user_category': user_category,
        'error': None
    }

def visualize_user_mask(image, user_mask):
    """Visualize user's mask overlay on image (NO ground truth comparison)"""
    if image is None or user_mask is None:
        return None
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    overlay = image.copy()
    # Show user mask in semi-transparent blue
    overlay[user_mask > 0] = overlay[user_mask > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
    
    return overlay.astype(np.uint8)

def visualize_comparison(image, pred_mask, gt_mask):
    """Visualize prediction vs ground truth (for final evaluation)"""
    if image is None or pred_mask is None or gt_mask is None:
        return None
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    overlay = image.copy()
    
    # True Positive (green)
    tp = np.logical_and(pred_mask, gt_mask)
    overlay[tp] = overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # False Positive (red)
    fp = np.logical_and(pred_mask, ~gt_mask.astype(bool))
    overlay[fp] = overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # False Negative (blue)
    fn = np.logical_and(~pred_mask.astype(bool), gt_mask)
    overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    return overlay.astype(np.uint8)

# Initialize session state
if 'seg_current_question' not in st.session_state:
    st.session_state.seg_current_question = 0
    st.session_state.seg_uploaded_jsons = {}  # {question_id: json_data}
    st.session_state.seg_test_started = False
    st.session_state.seg_test_completed = False

def start_seg_test():
    st.session_state.seg_test_started = True
    st.session_state.seg_current_question = 0
    st.session_state.seg_uploaded_jsons = {}
    st.session_state.seg_test_completed = False

def go_to_question(q_idx):
    st.session_state.seg_current_question = q_idx

def submit_all_answers():
    """Final submission - evaluate all answers"""
    st.session_state.seg_test_completed = True

def restart_seg_test():
    st.session_state.seg_current_question = 0
    st.session_state.seg_uploaded_jsons = {}
    st.session_state.seg_test_started = False
    st.session_state.seg_test_completed = False

# Main App
st.set_page_config(page_title="Segmentation Annotation Test", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .sidebar-question {
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        cursor: pointer;
    }
    .sidebar-question:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Welcome Screen
if not st.session_state.seg_test_started:
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Part 2: Segmentation Annotation Test</h1>
        <p style="font-size: 18px;">Upload your COCO format annotations for evaluation</p>
        <p style="font-size: 16px; opacity: 0.9;">7 Questions | IoU ≥ 0.9 Required | 100% Pass Rate</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ### 📋 Instructions
    - Navigate freely between questions using the sidebar
    - For each image, upload a **COCO format JSON file** with your annotations
    - Your JSON can contain **multiple objects** (multiple annotations)
    - The system will automatically find the **best matching annotation** by IoU
    - See your mask visualization **instantly** upon upload
    - To pass each question:
      - IoU must be ≥ 0.9
      - Category must match exactly
    - Click **"Submit All Answers"** when ready for final evaluation
    - You must pass **ALL 7 questions** to complete the test
    """)
    
    st.markdown("### 📄 Expected JSON Format")
    st.code("""
{
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [[x1, y1, x2, y2, x3, y3, ...]],
            "area": 1234,
            "bbox": [x, y, width, height],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }
    ]
}
    """, language="json")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Start Test", on_click=start_seg_test, type="primary", width="stretch")

# Test in Progress
elif st.session_state.seg_test_started and not st.session_state.seg_test_completed:
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📑 Question Navigation")
        
        # Progress overview
        completed_count = len(st.session_state.seg_uploaded_jsons)
        st.metric("Progress", f"{completed_count}/{len(SEGMENTATION_QUESTIONS)}")
        
        st.markdown("---")
        
        # Question list
        for idx, q in enumerate(SEGMENTATION_QUESTIONS):
            is_current = (idx == st.session_state.seg_current_question)
            has_upload = q['id'] in st.session_state.seg_uploaded_jsons
            
            status = "✅" if has_upload else "⭕"
            button_type = "primary" if is_current else "secondary"
            
            if st.button(
                f"{status} Question {q['id']}", 
                key=f"nav_{idx}",
                type=button_type,
                disabled=is_current,
                width="stretch"
            ):
                go_to_question(idx)
        
        st.markdown("---")
        
        # Submit button in sidebar
        if completed_count == len(SEGMENTATION_QUESTIONS):
            st.success(f"✅ All {len(SEGMENTATION_QUESTIONS)} questions completed!")
            if st.button("📤 Submit All Answers", type="primary", width="stretch"):
                submit_all_answers()
        else:
            st.warning(f"⚠️ {len(SEGMENTATION_QUESTIONS) - completed_count} questions remaining")
    
    # Main content
    q = SEGMENTATION_QUESTIONS[st.session_state.seg_current_question]
    q_id = q['id']
    
    # Question header
    st.markdown(f"""
    <div class="question-box">
        <h2>Question {q['id']} of {len(SEGMENTATION_QUESTIONS)}</h2>
        <p style="font-size: 14px; opacity: 0.9;">Upload your annotation JSON to see instant visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Test Image")
        try:
            image = Image.open(q['image_path'])
            st.image(image, width="stretch")
            
            # Download button
            import io
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="⬇️ Download Image",
                data=buf,
                file_name=f"question_{q['id']}_image.png",
                mime="image/png",
                width="stretch"
            )
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.subheader("📤 Upload Annotation")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            key=f"upload_{q_id}",
            help="Upload COCO format annotation JSON"
        )
        
        if uploaded_file is not None:
            try:
                # Parse JSON
                user_json = json.load(uploaded_file)
                
                # Convert LabelMe to COCO format if needed
                user_json = convert_labelme_to_coco(user_json)
                
                # Save to session state
                st.session_state.seg_uploaded_jsons[q_id] = user_json
                
                st.success("✅ File uploaded successfully!")
                
                # Show format detection
                if 'shapes' in json.loads(uploaded_file.getvalue().decode()):
                    st.info("📝 Detected LabelMe format - automatically converted to COCO format")
                
                with st.expander("📄 View uploaded JSON"):
                    st.json(user_json)
                
                num_annotations = len(user_json.get('annotations', []))
                st.info(f"📊 Found {num_annotations} annotation(s) in your file")
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid JSON.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
            
        elif q_id in st.session_state.seg_uploaded_jsons:
            st.success("✅ JSON already uploaded for this question")
            
            with st.expander("📄 View uploaded JSON"):
                st.json(st.session_state.seg_uploaded_jsons[q_id])
        
    # Instant Visualization Section
    if q_id in st.session_state.seg_uploaded_jsons:
        st.markdown("---")
        st.subheader("🎨 Your Annotation Visualization")
        st.caption("This shows YOUR segmentation mask overlaid on the image (blue). Evaluation will happen after final submission.")
        
        try:
            user_json = st.session_state.seg_uploaded_jsons[q_id]
            image_np = np.array(Image.open(q['image_path']))
            
            # Create visualization for all annotations or best one
            if 'annotations' in user_json and len(user_json['annotations']) > 0:
                
                # Combine all masks
                combined_mask = np.zeros(q['ground_truth']['image_shape'], dtype=np.uint8)
                for ann in user_json['annotations']:
                    try:
                        mask = polygon_to_mask(ann['segmentation'], q['ground_truth']['image_shape'])
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                    except:
                        continue
                
                overlay = visualize_user_mask(image_np, combined_mask)
                
                if overlay is not None:
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.markdown("**Original Image**")
                        st.image(image_np, width="stretch")
                    
                    with col_viz2:
                        st.markdown("**Your Mask Overlay**")
                        st.image(overlay, width="stretch")
                    
                    st.info(f"💡 Showing {len(user_json['annotations'])} annotation(s) from your JSON")
                else:
                    st.warning("Could not generate visualization")
            else:
                st.warning("No annotations found in the JSON file")
                
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")

# Final Results Screen
elif st.session_state.seg_test_completed:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Final Evaluation Results</h1>
        <p style="font-size: 18px;">Comparing your annotations with ground truth</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Evaluate all submissions
    results = []
    score = 0
    
    for q in SEGMENTATION_QUESTIONS:
        q_id = q['id']
        
        if q_id in st.session_state.seg_uploaded_jsons:
            user_json = st.session_state.seg_uploaded_jsons[q_id]
            gt_mask = load_ground_truth(q['ground_truth'])
            
            if gt_mask is not None:
                result = evaluate_submission(
                    user_json,
                    gt_mask,
                    q['ground_truth']['category'],
                    q['ground_truth']['image_shape']
                )
                result['question_id'] = q_id
                result['question'] = q
                results.append(result)
                
                if result['passed']:
                    score += 1
        else:
            results.append({
                'question_id': q_id,
                'question': q,
                'passed': False,
                'iou': 0.0,
                'class_match': False,
                'error': 'No submission'
            })
    
    # Summary
    percentage = (score / len(SEGMENTATION_QUESTIONS)) * 100
    passed = score == len(SEGMENTATION_QUESTIONS)
    
    if passed:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
            <h1 style="font-size: 48px; margin-bottom: 20px;">🎉 PERFECT SCORE!</h1>
            <h2>Your Score: {score}/{len(SEGMENTATION_QUESTIONS)}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                Congratulations! You have completed the Segmentation Annotation Test successfully!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f44336 0%, #e57373 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
            <h1 style="font-size: 48px; margin-bottom: 20px;">❌ INCOMPLETE</h1>
            <h2>Your Score: {score}/{len(SEGMENTATION_QUESTIONS)}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                You must pass ALL questions. Please try again.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Results
    st.subheader("📋 Question-by-Question Results")
    
    for result in results:
        q = result['question']
        
        with st.expander(f"Question {result['question_id']}: {q['ground_truth']['category']} - {'✅ PASS' if result['passed'] else '❌ FAIL'}"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IoU Score", f"{result['iou']:.4f}")
            
            with col2:
                st.metric("Category Match", "✅" if result['class_match'] else "❌")
                if not result['class_match'] and 'user_category' in result:
                    st.caption(f"Expected: {q['ground_truth']['category']}")
                    st.caption(f"Got: {result.get('user_category', 'unknown')}")
            
            with col3:
                st.metric("Result", "✅ PASS" if result['passed'] else "❌ FAIL")
            
            # Visualization with ground truth comparison
            if result.get('matched_mask') is not None:
                st.markdown("---")
                st.markdown("**Ground Truth Comparison**")
                st.caption("🟢 Green: Correct | 🔴 Red: False Positive | 🔵 Blue: False Negative")
                
                try:
                    image_np = np.array(Image.open(q['image_path']))
                    gt_mask = load_ground_truth(q['ground_truth'])
                    overlay = visualize_comparison(image_np, result['matched_mask'], gt_mask)
                    
                    if overlay is not None:
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.markdown("**Original**")
                            st.image(image_np, width="stretch")
                        
                        with col_b:
                            st.markdown("**Your Mask**")
                            user_viz = visualize_user_mask(image_np, result['matched_mask'])
                            st.image(user_viz, width="stretch")
                        
                        with col_c:
                            st.markdown("**Comparison**")
                            st.image(overlay, width="stretch")
                except Exception as e:
                    st.warning(f"Could not generate visualization: {str(e)}")
            
            elif result.get('error'):
                st.error(f"Error: {result['error']}")
    
    # Summary Table
    st.markdown("---")
    st.subheader("📊 Summary Table")
    
    summary_data = []
    for result in results:
        summary_data.append({
            "Question": result['question_id'],
            "Category": result['question']['ground_truth']['category'],
            "IoU": f"{result['iou']:.4f}" if result['iou'] > 0 else "N/A",
            "Category Match": "✅" if result.get('class_match') else "❌",
            "Result": "✅ PASS" if result['passed'] else "❌ FAIL"
        })
    
    st.dataframe(summary_data, width=None)
    
    # Restart button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("🔄 Restart Test", on_click=restart_seg_test, type="primary", width="stretch")