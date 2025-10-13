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
    # {
    #     "id": 2,
    #     "image_path": "./assets/ILSVRC2012_val_00007612.JPEG",
    #     "ground_truth": {
    #         "type": "mask",
    #         "mask_path": "./assets/ILSVRC2012_val_00007612_gt.png",
    #         "category": "dog",
    #         "image_shape": (480, 640)
    #     }
    # },
    # {
    #     "id": 3,
    #     "image_path": "./assets/ILSVRC2012_val_00013722.JPEG",
    #     "ground_truth": {
    #         "type": "mask",
    #         "mask_path": "./assets/ILSVRC2012_val_00013722_gt.png",
    #         "category": "car",
    #         "image_shape": (480, 640)
    #     }
    # },
    # {
    #     "id": 4,
    #     "image_path": "./assets/ILSVRC2012_val_00021098.JPEG",
    #     "ground_truth": {
    #         "type": "mask",
    #         "mask_path": "./assets/ILSVRC2012_val_00021098_gt.png",
    #         "category": "cat",
    #         "image_shape": (480, 640)
    #     }
    # },
    # {
    #     "id": 5,
    #     "image_path": "./assets/ILSVRC2012_val_00030960.JPEG",
    #     "ground_truth": {
    #         "type": "mask",
    #         "mask_path": "./assets/ILSVRC2012_val_00030960_gt.png",
    #         "category": "chair",
    #         "image_shape": (480, 640)
    #     }
    # },
    # {
    #     "id": 6,
    #     "image_path": "./assets/ILSVRC2012_val_00035508.JPEG",
    #     "ground_truth": {
    #         "type": "mask",
    #         "mask_path": "./assets/ILSVRC2012_val_00035508_gt.png",
    #         "category": "bicycle",
    #         "image_shape": (480, 640)
    #     }
    # },
    # {
    #     "id": 7,
    #     "image_path": "./assets/ILSVRC2012_val_00045093.JPEG",
    #     "ground_truth": {
    #         "type": "mask",
    #         "mask_path": "./assets/ILSVRC2012_val_00045093_gt.png",
    #         "category": "bottle",
    #         "image_shape": (480, 640)
    #     }
    # }
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
def convert_labelme_to_coco(labelme_json):
    """Convert LabelMe format to COCO format - merge shapes with same group_id"""
    if 'shapes' in labelme_json:
        coco_json = {
            'annotations': [],
            'categories': []
        }
        
        category_map = {}
        category_id = 1
        
        # Group shapes by group_id and label
        grouped_shapes = {}
        for shape in labelme_json['shapes']:
            label = shape.get('label', 'unknown')
            group_id = shape.get('group_id', None)
            key = (label, group_id if group_id is not None else id(shape))
            
            if key not in grouped_shapes:
                grouped_shapes[key] = []
            grouped_shapes[key].append(shape)
        
        # Convert each group to one annotation
        ann_id = 1
        for (label, group_id), shapes in grouped_shapes.items():
            if label not in category_map:
                category_map[label] = category_id
                coco_json['categories'].append({
                    'id': category_id,
                    'name': label,
                    'supercategory': label
                })
                category_id += 1
            
            all_polygons = []
            all_xs = []
            all_ys = []
            
            for shape in shapes:
                points = shape.get('points', [])
                if len(points) >= 3:
                    flat_points = []
                    for point in points:
                        flat_points.extend([float(point[0]), float(point[1])])
                        all_xs.append(point[0])
                        all_ys.append(point[1])
                    all_polygons.append(flat_points)
            
            if all_polygons and all_xs and all_ys:
                x_min, x_max = min(all_xs), max(all_xs)
                y_min, y_max = min(all_ys), max(all_ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = (x_max - x_min) * (y_max - y_min)
                
                coco_json['annotations'].append({
                    'id': ann_id,
                    'image_id': 1,
                    'category_id': category_map[label],
                    'segmentation': all_polygons,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                })
                ann_id += 1
        
        return coco_json
    else:
        return labelme_json

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
        mask = Image.open(file_path)
        mask_array = np.array(mask)
        
        # Handle RGB/RGBA masks (like red object on black background)
        if len(mask_array.shape) == 3:  # RGB or RGBA
            # Sum across color channels - any non-zero pixel is part of the mask
            binary_mask = (mask_array.sum(axis=2) > 0).astype(np.uint8)
        else:  # Grayscale
            binary_mask = (mask_array > 127).astype(np.uint8)
        
        return binary_mask
        
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
    """Get category name from category_id"""
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
    """Evaluate user submission"""
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
    
    actual_gt_shape = ground_truth_mask.shape
    
    for ann in user_json['annotations']:
        try:
            user_mask = polygon_to_mask(ann['segmentation'], actual_gt_shape)
            iou = calculate_iou(user_mask, ground_truth_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_annotation = ann
                best_mask = user_mask
        except Exception as e:
            st.warning(f"Error processing annotation: {str(e)}")
            continue
    
    if best_annotation is None:
        return {
            'passed': False,
            'iou': 0.0,
            'class_match': False,
            'matched_annotation': None,
            'matched_mask': None,
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
    """Visualize user's mask overlay"""
    if image is None or user_mask is None:
        return None
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    overlay = image.copy()
    overlay[user_mask > 0] = overlay[user_mask > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
    
    return overlay.astype(np.uint8)

def visualize_comparison(image, pred_mask, gt_mask):
    """Visualize prediction vs ground truth"""
    if image is None or pred_mask is None or gt_mask is None:
        return None
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    overlay = image.copy()
    
    tp = np.logical_and(pred_mask, gt_mask)
    overlay[tp] = overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    fp = np.logical_and(pred_mask, ~gt_mask.astype(bool))
    overlay[fp] = overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    fn = np.logical_and(~pred_mask.astype(bool), gt_mask)
    overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    return overlay.astype(np.uint8)

# Main App
st.set_page_config(page_title="Segmentation Annotation Test", page_icon="🎯", layout="wide")

# Initialize session state
if 'seg_current_question' not in st.session_state:
    st.session_state.seg_current_question = 0
if 'seg_uploaded_jsons' not in st.session_state:
    st.session_state.seg_uploaded_jsons = {}
if 'seg_test_started' not in st.session_state:
    st.session_state.seg_test_started = False
if 'seg_test_completed' not in st.session_state:
    st.session_state.seg_test_completed = False

def start_seg_test():
    st.session_state.seg_test_started = True
    st.session_state.seg_current_question = 0
    st.session_state.seg_uploaded_jsons = {}
    st.session_state.seg_test_completed = False

def go_to_question(q_idx):
    st.session_state.seg_current_question = q_idx

def submit_all_answers():
    st.session_state.seg_test_completed = True

def restart_seg_test():
    st.session_state.seg_current_question = 0
    st.session_state.seg_uploaded_jsons = {}
    st.session_state.seg_test_started = False
    st.session_state.seg_test_completed = False

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
</style>
""", unsafe_allow_html=True)

# Welcome Screen
if not st.session_state.seg_test_started:
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Segmentation Annotation Test</h1>
        <p style="font-size: 18px;">Upload your COCO format annotations for evaluation</p>
        <p style="font-size: 16px; opacity: 0.9;">1 Question | IoU ≥ 0.9 Required</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ### 📋 Instructions
    - Navigate freely between questions using the sidebar
    - Upload a **COCO or LabelMe format JSON file**
    - See your mask visualization **instantly** upon upload
    - Click **"Submit All Answers"** for final evaluation
    - To pass: IoU ≥ 0.9 AND category must match
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Start Test", on_click=start_seg_test, type="primary", width="stretch")

# Test in Progress
elif st.session_state.seg_test_started and not st.session_state.seg_test_completed:
    
    # Safety check
    if st.session_state.seg_current_question >= len(SEGMENTATION_QUESTIONS):
        st.session_state.seg_current_question = 0
        st.rerun()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📑 Question Navigation")
        
        completed_count = len(st.session_state.seg_uploaded_jsons)
        st.metric("Progress", f"{completed_count}/{len(SEGMENTATION_QUESTIONS)}")
        
        st.markdown("---")
        
        for idx, q in enumerate(SEGMENTATION_QUESTIONS):
            is_current = (idx == st.session_state.seg_current_question)
            has_upload = q['id'] in st.session_state.seg_uploaded_jsons
            
            status = "✅" if has_upload else "⭕"
            button_type = "primary" if is_current else "secondary"
            
            if st.button(
                f"{status} Question {q['id']}: {q['ground_truth']['category']}", 
                key=f"nav_{idx}",
                type=button_type,
                disabled=is_current,
                width="stretch"
            ):
                go_to_question(idx)
        
        st.markdown("---")
        
        if completed_count == len(SEGMENTATION_QUESTIONS):
            st.success(f"✅ All questions completed!")
            if st.button("📤 Submit All Answers", type="primary", width="stretch"):
                submit_all_answers()
        else:
            st.warning(f"⚠️ {len(SEGMENTATION_QUESTIONS) - completed_count} remaining")
    
    # Main content
    q = SEGMENTATION_QUESTIONS[st.session_state.seg_current_question]
    q_id = q['id']
    
    st.markdown(f"""
    <div class="question-box">
        <h2>Question {q['id']} of {len(SEGMENTATION_QUESTIONS)}</h2>
        <p style="font-size: 16px;">Expected Category: <strong>{q['ground_truth']['category']}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Test Image")
        try:
            image = Image.open(q['image_path'])
            image_np = np.array(image)
            
            display_image = image_np
            if q_id in st.session_state.seg_uploaded_jsons:
                try:
                    user_json = st.session_state.seg_uploaded_jsons[q_id]
                    actual_shape = image_np.shape[:2]
                    
                    if 'annotations' in user_json and len(user_json['annotations']) > 0:
                        combined_mask = np.zeros(actual_shape, dtype=np.uint8)
                        for ann in user_json['annotations']:
                            try:
                                mask = polygon_to_mask(ann['segmentation'], actual_shape)
                                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                            except:
                                continue
                        
                        overlay = visualize_user_mask(image_np, combined_mask)
                        if overlay is not None:
                            display_image = overlay
                            st.caption("🎨 Showing your annotation overlay")
                except:
                    pass
            
            st.image(display_image, width="stretch")
            
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
            
            # Ground Truth Visualization
            with st.expander("🔍 View Ground Truth Mask (Debug)"):
                try:
                    gt_mask = load_ground_truth(q['ground_truth'])
                    if gt_mask is not None:
                        st.success(f"✅ GT Mask loaded: shape={gt_mask.shape}, unique values={np.unique(gt_mask)}")
                        st.text(f"GT pixels (1s): {gt_mask.sum()}, Total: {gt_mask.size}")
                        
                        # Show GT mask overlay
                        gt_overlay = visualize_user_mask(image_np, gt_mask)
                        if gt_overlay is not None:
                            st.image(gt_overlay, caption="Ground Truth Mask Overlay", width="stretch")
                        
                        # Show raw mask
                        st.image(gt_mask * 255, caption="Raw GT Mask", width="stretch")

                    else:
                        st.error("❌ Failed to load GT mask")
                except Exception as e:
                    st.error(f"GT Debug Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    with col2:
        st.subheader("📤 Upload Annotation")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            key=f"upload_{q_id}",
            help="Upload COCO or LabelMe format JSON"
        )
        
        if uploaded_file is not None:
            try:
                user_json = json.load(uploaded_file)
                user_json = convert_labelme_to_coco(user_json)
                st.session_state.seg_uploaded_jsons[q_id] = user_json
                
                st.success("✅ File uploaded successfully!")
                
                num_annotations = len(user_json.get('annotations', []))
                st.info(f"📊 Found {num_annotations} annotation(s)")
                
                with st.expander("📄 View JSON"):
                    st.json(user_json)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        elif q_id in st.session_state.seg_uploaded_jsons:
            st.success("✅ JSON already uploaded")
            
            with st.expander("📄 View JSON"):
                st.json(st.session_state.seg_uploaded_jsons[q_id])

# Final Results
elif st.session_state.seg_test_completed:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Final Evaluation Results</h1>
    </div>
    """, unsafe_allow_html=True)
    
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
                'matched_mask': None,
                'error': 'No submission'
            })
    
    percentage = (score / len(SEGMENTATION_QUESTIONS)) * 100
    passed = score == len(SEGMENTATION_QUESTIONS)
    
    if passed:
        st.success(f"🎉 PERFECT SCORE! {score}/{len(SEGMENTATION_QUESTIONS)} ({percentage:.1f}%)")
    else:
        st.error(f"❌ Score: {score}/{len(SEGMENTATION_QUESTIONS)} ({percentage:.1f}%)")
    
    st.subheader("📋 Detailed Results")
    
    for result in results:
        q = result['question']
        
        with st.expander(f"Question {result['question_id']}: {q['ground_truth']['category']} - {'✅ PASS' if result['passed'] else '❌ FAIL'}"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IoU", f"{result['iou']:.4f}")
            
            with col2:
                st.metric("Category", "✅" if result['class_match'] else "❌")
                if 'user_category' in result:
                    st.caption(f"Expected: {q['ground_truth']['category']}")
                    st.caption(f"Got: {result.get('user_category', 'N/A')}")
            
            with col3:
                st.metric("Result", "✅ PASS" if result['passed'] else "❌ FAIL")
            
            if result.get('matched_mask') is not None:
                st.markdown("---")
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
                    st.warning(f"Visualization error: {str(e)}")
    
    st.markdown("---")
    st.subheader("📊 Summary")
    
    summary_data = []
    for result in results:
        summary_data.append({
            "Question": result['question_id'],
            "Category": result['question']['ground_truth']['category'],
            "IoU": f"{result['iou']:.4f}" if result['iou'] > 0 else "N/A",
            "Match": "✅" if result.get('class_match') else "❌",
            "Result": "✅ PASS" if result['passed'] else "❌ FAIL"
        })
    
    st.dataframe(summary_data, width="stretch")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("🔄 Restart", on_click=restart_seg_test, type="primary", width="stretch")