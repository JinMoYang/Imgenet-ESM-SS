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

# Helper Functions
def polygon_to_mask(segmentation, image_shape):
    """Convert COCO polygon format to binary mask"""
    height, width = image_shape
    mask = Image.new('L', (width, height), 0)
    
    for polygon in segmentation:
        # Flatten polygon coordinates
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
    # Fallback to standard COCO categories
    return COCO_CATEGORIES.get(category_id, "unknown")

def load_ground_truth(gt_data):
    """Load ground truth mask"""
    if gt_data['type'] == 'mask':
        return load_png_mask(gt_data['mask_path'])
    elif gt_data['type'] == 'json':
        with open(gt_data['json_path'], 'r') as f:
            gt_json = json.load(f)
            ann = gt_json['annotations'][0]  # Only one annotation
            return polygon_to_mask(ann['segmentation'], gt_data['image_shape'])
    return None

def evaluate_submission(user_json, ground_truth_mask, gt_category, image_shape):
    """
    Evaluate user submission
    - Find best matching annotation by IoU
    - Check if IoU >= 0.9 and category matches
    """
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
    
    # Find best matching annotation
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
    
    # Check category match
    user_category = get_category_name(best_annotation['category_id'], user_json)
    class_match = (user_category.lower() == gt_category.lower())
    
    # Pass if IoU >= 0.9 AND category matches
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

def visualize_comparison(image, pred_mask, gt_mask):
    """Visualize prediction vs ground truth"""
    if image is None or pred_mask is None or gt_mask is None:
        return None
    
    # Convert image to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    overlay = image.copy()
    
    # True Positive (green) - intersection
    tp = np.logical_and(pred_mask, gt_mask)
    overlay[tp] = overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # False Positive (red) - pred only
    fp = np.logical_and(pred_mask, ~gt_mask.astype(bool))
    overlay[fp] = overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # False Negative (blue) - gt only
    fn = np.logical_and(~pred_mask.astype(bool), gt_mask)
    overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    return overlay.astype(np.uint8)

# Initialize session state
if 'seg_current_question' not in st.session_state:
    st.session_state.seg_current_question = 0
    st.session_state.seg_results = []
    st.session_state.seg_score = 0
    st.session_state.seg_test_started = False
    st.session_state.seg_test_completed = False
    st.session_state.seg_uploaded_file = None
    st.session_state.seg_show_feedback = False

def start_seg_test():
    st.session_state.seg_test_started = True
    st.session_state.seg_current_question = 0
    st.session_state.seg_results = []
    st.session_state.seg_score = 0
    st.session_state.seg_test_completed = False
    st.session_state.seg_uploaded_file = None
    st.session_state.seg_show_feedback = False

def submit_annotation():
    if st.session_state.seg_uploaded_file is None:
        st.warning("⚠️ Please upload your annotation JSON file!")
        return
    
    st.session_state.seg_show_feedback = True

def next_seg_question():
    st.session_state.seg_current_question += 1
    st.session_state.seg_uploaded_file = None
    st.session_state.seg_show_feedback = False
    
    if st.session_state.seg_current_question >= len(SEGMENTATION_QUESTIONS):
        st.session_state.seg_test_completed = True

def restart_seg_test():
    st.session_state.seg_current_question = 0
    st.session_state.seg_results = []
    st.session_state.seg_score = 0
    st.session_state.seg_test_started = False
    st.session_state.seg_test_completed = False
    st.session_state.seg_uploaded_file = None
    st.session_state.seg_show_feedback = False

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
    - For each image, you will upload a **COCO format JSON file** with your annotations
    - Your JSON can contain **multiple objects** (multiple annotations)
    - The system will automatically find the **best matching annotation** by IoU
    - To pass each question:
      - IoU must be ≥ 0.9
      - Category must match exactly
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
        st.button("Start Test", on_click=start_seg_test, type="primary", width = 'stretch')

# Test in Progress
elif st.session_state.seg_test_started and not st.session_state.seg_test_completed:
    q = SEGMENTATION_QUESTIONS[st.session_state.seg_current_question]
    
    # Progress bar
    progress = st.session_state.seg_current_question / len(SEGMENTATION_QUESTIONS)
    st.progress(progress)
    st.write(f"**Question {st.session_state.seg_current_question + 1} / {len(SEGMENTATION_QUESTIONS)}**")
    
    # Question header
    st.markdown(f"""
    <div class="question-box">
        <h2>Question {q['id']}</h2>
        <p style="font-size: 16px;">Expected Category: <strong>{q['ground_truth']['category']}</strong></p>
        <p style="font-size: 14px; opacity: 0.9;">Upload your annotation JSON file for this image</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Test Image")
        try:
            image = Image.open(q['image_path'])
            st.image(image, width = 'stretch')
            
            # Download button for the image
            import io
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="⬇️ Download Image",
                data=buf,
                file_name=f"question_{q['id']}_image.png",
                mime="image/png",
                width = 'stretch'
            )
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.info("📁 Please ensure test images are in the correct path")
    
    with col2:
        st.subheader("Upload Your Annotation")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            key=f"upload_{st.session_state.seg_current_question}",
            help="Upload COCO format annotation JSON"
        )
        
        if uploaded_file is not None:
            st.session_state.seg_uploaded_file = uploaded_file
            
            try:
                # Parse JSON
                user_json = json.load(uploaded_file)
                
                # Show preview
                st.success("✅ File uploaded successfully!")
                
                with st.expander("📄 View uploaded JSON"):
                    st.json(user_json)
                
                # Show annotations count
                num_annotations = len(user_json.get('annotations', []))
                st.info(f"📊 Found {num_annotations} annotation(s) in your file")
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid JSON.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Feedback section
    if st.session_state.seg_show_feedback and st.session_state.seg_uploaded_file is not None:
        st.markdown("---")
        st.subheader("📊 Evaluation Results")
        
        try:
            # Load user JSON
            st.session_state.seg_uploaded_file.seek(0)  # Reset file pointer
            user_json = json.load(st.session_state.seg_uploaded_file)
            
            # Load ground truth
            gt_mask = load_ground_truth(q['ground_truth'])
            
            if gt_mask is None:
                st.error("❌ Error loading ground truth data")
            else:
                # Evaluate
                result = evaluate_submission(
                    user_json, 
                    gt_mask, 
                    q['ground_truth']['category'],
                    q['ground_truth']['image_shape']
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("IoU Score", f"{result['iou']:.4f}")
                
                with col2:
                    st.metric("Category Match", "✅" if result['class_match'] else "❌")
                    if not result['class_match']:
                        st.caption(f"Expected: {q['ground_truth']['category']}")
                        st.caption(f"Got: {result.get('user_category', 'unknown')}")
                
                with col3:
                    st.metric("Result", "PASS ✅" if result['passed'] else "FAIL ❌")
                
                # Detailed feedback
                if result['passed']:
                    st.markdown("""
                    <div class="success-box">
                        <strong style="color: #155724;">✅ Excellent! Your annotation passed!</strong><br>
                        <span style="color: #155724;">IoU ≥ 0.9 and category matches correctly.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    reasons = []
                    if result['iou'] < 0.9:
                        reasons.append(f"IoU too low: {result['iou']:.4f} < 0.9")
                    if not result['class_match']:
                        reasons.append(f"Category mismatch: expected '{q['ground_truth']['category']}', got '{result.get('user_category', 'unknown')}'")
                    
                    st.markdown(f"""
                    <div class="error-box">
                        <strong style="color: #721c24;">❌ Failed!</strong><br>
                        <span style="color: #721c24;">{' | '.join(reasons)}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualization
                if result['matched_mask'] is not None:
                    st.subheader("🔍 Visualization")
                    st.caption("Green: Correct overlap | Red: False Positive | Blue: False Negative")
                    
                    try:
                        image_np = np.array(Image.open(q['image_path']))
                        overlay = visualize_comparison(image_np, result['matched_mask'], gt_mask)
                        if overlay is not None:
                            st.image(overlay, width = 'stretch')
                    except Exception as e:
                        st.warning(f"Could not generate visualization: {str(e)}")
                
                # Save result
                st.session_state.seg_results.append(result)
                if result['passed']:
                    st.session_state.seg_score += 1
                
                # Next button
                st.button("Next Question →", on_click=next_seg_question, type="primary")
                
        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
    
    elif not st.session_state.seg_show_feedback:
        st.button("Submit Annotation", on_click=submit_annotation, type="primary", disabled=(st.session_state.seg_uploaded_file is None))

# Results Screen
elif st.session_state.seg_test_completed:
    percentage = (st.session_state.seg_score / len(SEGMENTATION_QUESTIONS)) * 100
    passed = st.session_state.seg_score == len(SEGMENTATION_QUESTIONS)
    
    # Result summary
    if passed:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
            <h1 style="font-size: 48px; margin-bottom: 20px;">🎉 PERFECT SCORE!</h1>
            <h2>Your Score: {st.session_state.seg_score}/{len(SEGMENTATION_QUESTIONS)}</h2>
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
            <h2>Your Score: {st.session_state.seg_score}/{len(SEGMENTATION_QUESTIONS)}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                You must pass ALL questions. Please try again.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    st.subheader("📊 Detailed Results")
    
    results_data = []
    for i, result in enumerate(st.session_state.seg_results):
        results_data.append({
            "Question": i + 1,
            "IoU": f"{result['iou']:.4f}",
            "Category Match": "✅" if result['class_match'] else "❌",
            "Result": "✅ PASS" if result['passed'] else "❌ FAIL"
        })
    
    st.dataframe(results_data, width = 'stretch')
    
    # Restart button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Restart Test", on_click=restart_seg_test, type="primary", width = 'stretch')