"""
Objectness QA Test - Merged Annotations Review Application

This Streamlit app allows reviewers to assess merged image annotations from multiple annotators.
Reviewers can evaluate objectness quality, suggest object additions/subtractions, and provide
feedback that is automatically saved to Google Sheets.

DIRECTORY STRUCTURE REQUIREMENTS:
    Tests/
    ├── objectness_qa_test.py
    └── assets/
        ├── original/                      # Place all IMAGE files here (.png, .jpg, .JPEG, etc.)
        │   ├── image1.png
        │   ├── image2.JPEG
        │   └── ...
        └── merged_results/
            └── merged_annotations/        # Place all JSON annotation files here
                ├── image1.json            # Must match image filename (image1.png -> image1.json)
                ├── image2.json
                └── ...

HOW IT WORKS:
    1. App scans ./assets/original/ for all image files
    2. Filters to only show images that have matching JSON files in ./assets/merged_results/merged_annotations/
    3. Displays images with polygon overlays from annotations
    4. Reviewers answer questions and save responses (auto-uploaded to Google Sheets)

USAGE:
    streamlit run objectness_qa_test.py
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path

# Configuration
IMAGE_DIR = "./assets/original"
MERGED_ANNOTATIONS_DIR = "./assets/merged_results/merged_annotations"

def load_merged_annotation(image_name):
    """Load merged annotation JSON for a given image."""
    json_name = Path(image_name).stem + '.json'
    json_path = os.path.join(MERGED_ANNOTATIONS_DIR, json_name)

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading annotation: {e}")
        return None

def draw_polygons_on_image(image, annotation_data):
    """Draw all merged polygons on an image."""
    # Create a copy to draw on
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Define colors for polygons
    colors = ['#00FF00', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8C00']

    if annotation_data and 'shapes' in annotation_data:
        for idx, shape in enumerate(annotation_data['shapes']):
            if shape['shape_type'] == 'polygon':
                points = shape['points']
                label = shape.get('label', 'unknown')

                # Convert points to tuples
                polygon_points = [tuple(p) for p in points]

                # Choose color
                color = colors[idx % len(colors)]

                # Draw polygon outline (thick line)
                draw.polygon(polygon_points, outline=color, width=4)

                # Draw label near the first point
                if polygon_points:
                    text_pos = polygon_points[0]
                    # Try to load a font, fall back to default if not available
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    except:
                        font = ImageFont.load_default()

                    # Draw text background
                    bbox = draw.textbbox(text_pos, label, font=font)
                    draw.rectangle(bbox, fill='black')
                    draw.text(text_pos, label, fill=color, font=font)

    return img_copy

@st.cache_resource
def get_gsheet_connection():
    """Initialize Google Sheets connection and return spreadsheet"""
    try:
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1sPD3BKub_lITqwvo51SQ7C35AMMJGeD8T2FshSzaUT4/edit")
        return spreadsheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        return None

def get_or_create_reviewer_worksheet(spreadsheet, reviewer_name):
    """Get or create a worksheet for specific reviewer"""
    if not spreadsheet or not reviewer_name:
        return None
    
    try:
        worksheet_name = reviewer_name.strip()[:100]
        
        # Try to get existing worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            return worksheet
        except gspread.exceptions.WorksheetNotFound:
            # Create new worksheet
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=10)
            
            # Add headers
            headers = [
                'Timestamp',
                'Reviewer Name',
                'Image Filename',
                'Q1: Objectness Appropriate?',
                'Q2: Objects to ADD',
                'Q3: Objects to SUBTRACT',
                'Q4: Other Comments'
            ]
            worksheet.append_row(headers)
            
            # Format header row
            worksheet.format('A1:G1', {
                'textFormat': {'bold': True, 'fontSize': 11},
                'backgroundColor': {'red': 0.8, 'green': 0.9, 'blue': 1.0},
                'horizontalAlignment': 'CENTER',
                'wrapStrategy': 'WRAP'
            })
            
            # Freeze header row
            worksheet.freeze(rows=1)
            
            return worksheet
    except Exception as e:
        st.error(f"Error with worksheet: {str(e)}")
        return None

def append_single_review_to_sheet(reviewer_name, image_name, response):
    """Append a single review to Google Sheets immediately"""
    if not reviewer_name or not response:
        return False
    
    spreadsheet = get_gsheet_connection()
    if spreadsheet is None:
        return False
    
    worksheet = get_or_create_reviewer_worksheet(spreadsheet, reviewer_name)
    if worksheet is None:
        return False
    
    try:
        row = [
            response.get('timestamp', ''),
            reviewer_name,
            image_name,
            response.get('objectness_appropriate', ''),
            response.get('objects_to_add', ''),
            response.get('objects_to_subtract', ''),
            response.get('other_comments', '')
        ]
        worksheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {str(e)}")
        return False

def check_if_image_already_reviewed(reviewer_name, image_name):
    """Check if this image has already been saved to Google Sheets"""
    spreadsheet = get_gsheet_connection()
    if spreadsheet is None:
        return False
    
    try:
        worksheet = get_or_create_reviewer_worksheet(spreadsheet, reviewer_name)
        if worksheet is None:
            return False
        
        # Get all values from the image filename column (column C, index 3)
        all_values = worksheet.col_values(3)
        return image_name in all_values
    except:
        return False

def get_image_list():
    """Get list of images that have merged annotations (clean merges only)"""
    # Check if directories exist
    if not os.path.exists(IMAGE_DIR):
        return []
    if not os.path.exists(MERGED_ANNOTATIONS_DIR):
        return []

    # Get all image files from assets directory
    image_names = []
    for ext in ['*.JPEG', '*.jpeg', '*.jpg', '*.JPG', '*.png', '*.PNG']:
        image_files = glob.glob(os.path.join(IMAGE_DIR, ext))
        image_names.extend([os.path.basename(f) for f in image_files])

    # Filter: only keep images that have corresponding JSON annotations
    filtered_images = []
    for image_name in image_names:
        image_stem = Path(image_name).stem
        json_path = os.path.join(MERGED_ANNOTATIONS_DIR, image_stem + '.json')

        # Only include if JSON annotation exists
        if os.path.exists(json_path):
            filtered_images.append(image_name)

    filtered_images.sort()
    return filtered_images

# Initialize session state
if 'current_image_idx' not in st.session_state:
    st.session_state.current_image_idx = 0
    st.session_state.reviewer_name = ""
    st.session_state.responses = {}
    st.session_state.images = get_image_list()
    st.session_state.app_started = False
    st.session_state.saved_to_sheet = set()  # Track which images are saved to sheet

def start_app():
    """Start the review app and create worksheet"""
    if st.session_state.reviewer_name.strip():
        # Create worksheet when app starts
        spreadsheet = get_gsheet_connection()
        if spreadsheet:
            worksheet = get_or_create_reviewer_worksheet(spreadsheet, st.session_state.reviewer_name)
            if worksheet:
                st.session_state.app_started = True
                return True
    return False

def go_to_image(idx):
    """Navigate to specific image index"""
    if 0 <= idx < len(st.session_state.images):
        st.session_state.current_image_idx = idx
        load_saved_response()

def next_image():
    """Go to next image"""
    if st.session_state.current_image_idx < len(st.session_state.images) - 1:
        st.session_state.current_image_idx += 1
        load_saved_response()

def prev_image():
    """Go to previous image"""
    if st.session_state.current_image_idx > 0:
        st.session_state.current_image_idx -= 1
        load_saved_response()

def load_saved_response():
    """Load saved response for current image if exists"""
    current_image = st.session_state.images[st.session_state.current_image_idx]
    if current_image in st.session_state.responses:
        response = st.session_state.responses[current_image]
        st.session_state.objectness_appropriate = response.get('objectness_appropriate', '')
        st.session_state.objects_to_add = response.get('objects_to_add', '')
        st.session_state.objects_to_subtract = response.get('objects_to_subtract', '')
        st.session_state.other_comments = response.get('other_comments', '')
    else:
        st.session_state.objectness_appropriate = ''
        st.session_state.objects_to_add = ''
        st.session_state.objects_to_subtract = ''
        st.session_state.other_comments = ''

def save_current_response():
    """Save current response to session state AND Google Sheets (always creates new row)"""
    current_image = st.session_state.images[st.session_state.current_image_idx]
    
    # Prepare response data
    response_data = {
        'objectness_appropriate': st.session_state.get('objectness_appropriate', ''),
        'objects_to_add': st.session_state.get('objects_to_add', ''),
        'objects_to_subtract': st.session_state.get('objects_to_subtract', ''),
        'other_comments': st.session_state.get('other_comments', ''),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to session state (for UI display)
    st.session_state.responses[current_image] = response_data
    
    # Always save to Google Sheets (creates new row even if already exists)
    if st.session_state.reviewer_name:
        success = append_single_review_to_sheet(
            st.session_state.reviewer_name,
            current_image,
            response_data
        )
        if success:
            st.session_state.saved_to_sheet.add(current_image)
            return True
    return False

# Page config
st.set_page_config(
    page_title="Merged Annotations Review",
    page_icon="🔍",
    layout="wide"
)

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
    .image-container {
        border: 3px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .question-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .nav-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Welcome/Setup Page
if not st.session_state.app_started:
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Merged Annotations Review</h1>
        <p style="font-size: 18px;">Review Merged Annotations from Multiple Annotators</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📋 Setup Instructions")

    st.markdown("""
    ### How to Use This App

    **About Merged Annotations:**
    - This app displays images with merged annotations from 3 annotators
    - Only successfully merged images (no errors) are shown
    - Multiple objects per image are supported

    **Navigation:**
    - Use **Previous/Next** buttons to move between images
    - **Jump to Image** - Enter any image number to jump directly
    - Progress is tracked in the sidebar
    
    **Questions for Each Image:**
    1. **Is the objectness appropriate?** - Select Yes/No
    2. **Objects to add** - List any objects that should be annotated but aren't
    3. **Objects to subtract** - List any objects that shouldn't be annotated
    4. **Other comments** - Additional feedback or observations
    
    **Saving Your Work:**
    - Click "Save Response" to save the current image review
    - **Each save automatically uploads to Google Sheets**
    - Your worksheet is created when you start the app
    - All reviews are immediately backed up to the cloud
    """)

    st.markdown("---")

    # Check if images exist
    total_images = len(st.session_state.images)
    if total_images > 0:
        st.success(f"✅ Found {total_images} images ready for review")
    else:
        st.error(f"❌ No images found in {IMAGE_DIR} directory!")
        st.info(f"Please add JPEG images to the {IMAGE_DIR} directory before starting.")

    st.markdown("---")

    # Reviewer name input
    st.markdown("### 👤 Reviewer Information")
    reviewer_name_input = st.text_input(
        "Enter your name to begin:",
        value=st.session_state.reviewer_name,
        placeholder="Your full name",
        key="welcome_reviewer_input"
    )

    st.markdown("---")

    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        can_start = total_images > 0 and reviewer_name_input.strip() != ""

        if st.button(
            "🚀 Start Review",
            type="primary",
            use_container_width=True,
            disabled=not can_start
        ):
            st.session_state.reviewer_name = reviewer_name_input.strip()
            with st.spinner("Creating your worksheet..."):
                if start_app():
                    st.success("✅ Worksheet created successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to create worksheet. Please check your connection.")

        if not reviewer_name_input.strip() and total_images > 0:
            st.info("👆 Please enter your name above to start")

    st.stop()

# Main Review Interface
st.markdown("""
<div class="main-header">
    <h1>🔍 Merged Annotations Review</h1>
    <p style="font-size: 18px;">Review merged annotations and provide feedback</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Reviewer Information")
    st.info(f"👤 **{st.session_state.reviewer_name}**")

    st.divider()

    # Progress
    st.header("Progress")
    total_images = len(st.session_state.images)
    reviewed_count = len(st.session_state.responses)
    saved_count = len(st.session_state.saved_to_sheet)
    
    st.metric("Images Reviewed", f"{reviewed_count} / {total_images}")
    st.metric("Saved to Sheets", f"{saved_count} / {total_images}")

    if total_images > 0:
        progress = reviewed_count / total_images
        st.progress(progress)

    st.divider()

    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        1. **Review each image** and answer the questions
        2. **Click "Save Response"** - auto-uploads to Google Sheets
        3. **Navigate** using Prev/Next buttons
        4. Your worksheet: **{name}**
        
        **Auto-save:** Each time you click Save Response, 
        your review is immediately added to Google Sheets!
        """.format(name=st.session_state.reviewer_name))

# Check if images exist
if not st.session_state.images:
    st.error(f"❌ No images found in {IMAGE_DIR} directory!")
    st.stop()

# Main content area
current_idx = st.session_state.current_image_idx
total_images = len(st.session_state.images)
current_image = st.session_state.images[current_idx]

# Navigation controls
st.markdown('<div class="nav-box">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    if st.button("⬅️ Previous", disabled=(current_idx == 0)):
        prev_image()
        st.rerun()

with col2:
    if st.button("Next ➡️", disabled=(current_idx >= total_images - 1)):
        next_image()
        st.rerun()

with col3:
    jump_to = st.number_input(
        f"Image Number (1-{total_images}):",
        min_value=1,
        max_value=total_images,
        value=current_idx + 1,
        key="jump_input"
    )
    if st.button("Jump to Image"):
        go_to_image(jump_to - 1)
        st.rerun()

with col4:
    st.write(f"**{current_idx + 1} / {total_images}**")

with col5:
    if current_image in st.session_state.saved_to_sheet:
        st.success("✅ Saved")
    elif current_image in st.session_state.responses:
        st.warning("💾 Local")
    else:
        st.info("⏳ New")

st.markdown('</div>', unsafe_allow_html=True)

# Display original and annotated images side by side
st.subheader("📸 Image with Merged Annotations")
col_img1, col_img2 = st.columns([1, 1])

# Load the image
image_path = os.path.join(IMAGE_DIR, current_image)
merged_annotation = load_merged_annotation(current_image)

with col_img1:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("**Original**")

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=current_image, use_container_width=True)
    else:
        st.error(f"❌ Image not found: {image_path}")
        image = None

    st.markdown('</div>', unsafe_allow_html=True)

with col_img2:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("**With Merged Annotations**")

    if image is not None and merged_annotation is not None:
        # Draw polygons on image
        annotated_image = draw_polygons_on_image(image, merged_annotation)
        st.image(annotated_image, caption=f"{current_image} (annotated)", use_container_width=True)

        # Display annotation info
        num_objects = len(merged_annotation.get('shapes', []))
        labels = [shape['label'] for shape in merged_annotation.get('shapes', [])]
        st.info(f"🔍 Objects: {num_objects} | Labels: {', '.join(labels)}")
    elif merged_annotation is None:
        st.error(f"❌ Merged annotation not found")
    else:
        st.error(f"❌ Cannot display annotations")

    st.markdown('</div>', unsafe_allow_html=True)

# Questions section
st.markdown("---")
st.subheader("Review Questions")

col_qa = st.container()

with col_qa:
    # Load saved response if exists
    if current_image in st.session_state.responses:
        saved = st.session_state.responses[current_image]
    else:
        saved = {
            'objectness_appropriate': '',
            'objects_to_add': '',
            'objects_to_subtract': '',
            'other_comments': ''
        }

    # Q1
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    
    # Determine the index for radio button
    saved_value = saved.get('objectness_appropriate', '')
    if saved_value == "Yes":
        radio_index = 0
    elif saved_value == "No":
        radio_index = 1
    else:
        radio_index = None  # No selection (default)
    
    objectness = st.radio(
        "**1. Is the objectness appropriate?**",
        options=["Yes", "No"],
        index=radio_index,
        key="q1_objectness",
        horizontal=True
    )
    st.session_state.objectness_appropriate = objectness if objectness else ''
    st.markdown('</div>', unsafe_allow_html=True)

    # Q2
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.write("**2. If not, should we add objects?**")
    objects_add = st.text_area(
        "List objects to add:",
        value=saved['objects_to_add'],
        key="q2_add",
        height=100,
        placeholder="e.g., person, car, tree"
    )
    st.session_state.objects_to_add = objects_add
    st.markdown('</div>', unsafe_allow_html=True)

    # Q3
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.write("**3. If not, should we subtract objects?**")
    objects_subtract = st.text_area(
        "List objects to remove:",
        value=saved['objects_to_subtract'],
        key="q3_subtract",
        height=100,
        placeholder="e.g., background, noise"
    )
    st.session_state.objects_to_subtract = objects_subtract
    st.markdown('</div>', unsafe_allow_html=True)

    # Q4
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.write("**4. Any other comments?**")
    other = st.text_area(
        "Additional feedback:",
        value=saved['other_comments'],
        key="q4_other",
        height=100,
        placeholder="Any other observations..."
    )
    st.session_state.other_comments = other
    st.markdown('</div>', unsafe_allow_html=True)

    # Save button - always creates new row and moves to next image
    if st.button("💾 Save Response & Upload to Sheets", type="primary", use_container_width=True):
        with st.spinner("Saving to Google Sheets..."):
            if save_current_response():
                st.success("✅ Response saved!")
                # Auto-advance to next image if not at the end
                if st.session_state.current_image_idx < len(st.session_state.images) - 1:
                    st.session_state.current_image_idx += 1
                    load_saved_response()
                st.rerun()
            else:
                st.error("❌ Failed to save to Google Sheets. Please try again.")
                st.rerun()