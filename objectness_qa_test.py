import streamlit as st
from PIL import Image
import os
import glob
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# Configuration
IMAGE_DIR = "./assets/objectness_original"
IMAGE_DIR_2 = "./assets/objectness_checked"  # Second version directory

@st.cache_resource
def get_gsheet_connection():
    """Initialize Google Sheets connection and return spreadsheet"""
    try:
        # Get credentials from Streamlit secrets
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )

        client = gspread.authorize(credentials)

        # Open spreadsheet by URL
        # spreadsheet = client.open_by_url(st.secrets["spreadsheet_url"])
        spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1sPD3BKub_lITqwvo51SQ7C35AMMJGeD8T2FshSzaUT4/edit")

        return spreadsheet

    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        st.info("Make sure you've set up Service Account credentials in secrets")
        return None

def get_reviewer_worksheet(spreadsheet, reviewer_name):
    """Get or create a worksheet for specific reviewer with formatted headers"""
    if not spreadsheet or not reviewer_name:
        return None

    try:
        # Sanitize reviewer name for worksheet title (max 100 chars)
        worksheet_name = reviewer_name.strip()[:100]

        # Try to get existing worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            return worksheet
        except gspread.exceptions.WorksheetNotFound:
            # Create new worksheet for this reviewer
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=10)

            # Add descriptive headers with clear question formatting
            headers = [
                'Timestamp (When Reviewed)',
                'Reviewer Name',
                'Image Filename',
                'Q1: Is the objectness appropriate? (Yes/No)',
                'Q2: Objects to ADD (if missing from annotation)',
                'Q3: Objects to SUBTRACT (if incorrectly annotated)',
                'Q4: Other Comments / Additional Feedback'
            ]
            worksheet.append_row(headers)

            # Format the header row
            worksheet.format('A1:G1', {
                'textFormat': {'bold': True, 'fontSize': 11},
                'backgroundColor': {'red': 0.8, 'green': 0.9, 'blue': 1.0},
                'horizontalAlignment': 'CENTER',
                'wrapStrategy': 'WRAP'
            })

            # Set column widths for better readability
            worksheet.set_column_width('A', 180)  # Timestamp
            worksheet.set_column_width('B', 150)  # Reviewer
            worksheet.set_column_width('C', 200)  # Image name
            worksheet.set_column_width('D', 280)  # Q1
            worksheet.set_column_width('E', 320)  # Q2 - Objects to add
            worksheet.set_column_width('F', 320)  # Q3 - Objects to subtract
            worksheet.set_column_width('G', 350)  # Q4 - Comments

            # Freeze header row
            worksheet.freeze(rows=1)

            return worksheet

    except Exception as e:
        st.error(f"Error creating reviewer worksheet: {str(e)}")
        return None

def save_to_google_sheets(reviewer_name, responses):
    """Save all responses to Google Sheets in reviewer-specific worksheet"""
    if not reviewer_name:
        st.error("Please enter your name before submitting!")
        return False, 0

    if not responses:
        st.warning("No reviews to submit yet!")
        return False, 0

    # Get spreadsheet connection
    spreadsheet = get_gsheet_connection()
    if spreadsheet is None:
        return False, 0

    # Get or create reviewer-specific worksheet
    worksheet = get_reviewer_worksheet(spreadsheet, reviewer_name)
    if worksheet is None:
        return False, 0

    try:
        # Prepare rows to append
        rows_to_add = []
        for image_name, response in responses.items():
            row = [
                response['timestamp'],
                reviewer_name,
                image_name,
                response['objectness_appropriate'],
                response['objects_to_add'],
                response['objects_to_subtract'],
                response['other_comments']
            ]
            rows_to_add.append(row)

        # Append all rows at once
        if rows_to_add:
            worksheet.append_rows(rows_to_add)

        return True, len(rows_to_add)

    except Exception as e:
        st.error(f"Error saving to Google Sheets: {str(e)}")
        return False, 0

def get_image_list():
    """Get list of all JPEG images in the assets directory"""
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.JPEG"))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.JPG")))
    image_files.sort()
    return [os.path.basename(f) for f in image_files]

# Initialize session state
if 'current_image_idx' not in st.session_state:
    st.session_state.current_image_idx = 0
    st.session_state.reviewer_name = ""
    st.session_state.responses = {}
    st.session_state.images = get_image_list()
    st.session_state.app_started = False

def start_app():
    """Start the review app"""
    st.session_state.app_started = True

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
    """Save current response to session state"""
    current_image = st.session_state.images[st.session_state.current_image_idx]
    st.session_state.responses[current_image] = {
        'objectness_appropriate': st.session_state.get('objectness_appropriate', ''),
        'objects_to_add': st.session_state.get('objects_to_add', ''),
        'objects_to_subtract': st.session_state.get('objects_to_subtract', ''),
        'other_comments': st.session_state.get('other_comments', ''),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Page config
st.set_page_config(
    page_title="Objectness QA Test",
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
        <h1>🔍 Objectness QA Review</h1>
        <p style="font-size: 18px;">Welcome to the Image Annotation Review System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📋 Setup Instructions")

    st.markdown("""

    ### How to Use This App
    **Meaning of Stars:**
    - 🔴 Red stars : Clear COCO object
    - 🔵 Blue stars : Clear non-COCO object
    - 🟡 Yellow stars : Flagged object
    - 🟢 Green stars : isCrowd annotations
    - 🚫 No stars : false negative images

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
    - Responses auto-save when you navigate to another image
    - Click "Save Response" button to manually save
    - Your progress persists during the session
    
    **Submitting:**
    - Click "Submit All to Google Sheets" when done
    - All reviews automatically saved to your Google Sheet [ImageNet-ES Meta (CVPR 2026) --> 프로젝트 진행 --> Objectness QA Test 스프레드 시트]
    """)

    st.markdown("---")

    # Check if images exist
    total_images = len(st.session_state.images)
    if total_images > 0:
        st.success(f"✅ Found {total_images} images ready for review")
    else:
        st.error(f"❌ No images found in {IMAGE_DIR} directory!")
        st.info("Please add JPEG images to the ./assets/ directory before starting.")

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
        # Enable button only if name is entered and images exist
        can_start = total_images > 0 and reviewer_name_input.strip() != ""

        if st.button(
            "🚀 Start Review",
            type="primary",
            use_container_width=True,
            disabled=not can_start
        ):
            st.session_state.reviewer_name = reviewer_name_input.strip()
            start_app()
            st.rerun()

        if not reviewer_name_input.strip() and total_images > 0:
            st.info("👆 Please enter your name above to start")

    st.stop()

# Main Review Interface
st.markdown("""
<div class="main-header">
    <h1>🔍 Objectness QA Review</h1>
    <p style="font-size: 18px;">Review image annotations and provide feedback</p>
</div>
""", unsafe_allow_html=True)

# Reviewer name input (sidebar)
with st.sidebar:
    st.header("Reviewer Information")
    reviewer_name = st.text_input(
        "Your Name:",
        value=st.session_state.reviewer_name,
        key="reviewer_input"
    )
    if reviewer_name:
        st.session_state.reviewer_name = reviewer_name

    st.divider()

    # Progress
    st.header("Progress")
    total_images = len(st.session_state.images)
    reviewed_count = len(st.session_state.responses)
    st.metric("Images Reviewed", f"{reviewed_count} / {total_images}")

    if total_images > 0:
        progress = reviewed_count / total_images
        st.progress(progress)

    st.divider()

    # Submit options
    st.header("Submit Reviews")
    
    # Google Sheets submit
    if st.button("📤 Submit All to Google Sheets", type="primary", use_container_width=True):
        if reviewed_count == 0:
            st.warning("No reviews to submit yet!")
        else:
            with st.spinner("Submitting to Google Sheets..."):
                success, count = save_to_google_sheets(
                    st.session_state.reviewer_name,
                    st.session_state.responses
                )
                if success:
                    st.success(f"✅ Successfully submitted {count} reviews!")

    st.divider()

    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        1. **Enter your name** in the sidebar
        2. **Review each image** and answer the questions
        3. **Navigate** using Prev/Next buttons
        4. Responses are **auto-saved** as you type
        5. Click **Submit All to Google Sheets** when done

        **Questions:**
        - **Objectness Appropriate?** - Does annotation identify all objects correctly?
        - **Objects to Add** - Missing objects
        - **Objects to Subtract** - Incorrectly annotated objects
        - **Other Comments** - Additional feedback
        """)

# Check if images exist
if not st.session_state.images:
    st.error(f"❌ No images found in {IMAGE_DIR} directory!")
    st.info("Please add JPEG images to the ./assets/ directory.")
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
        save_current_response()
        prev_image()
        st.rerun()

with col2:
    if st.button("Next ➡️", disabled=(current_idx >= total_images - 1)):
        save_current_response()
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
        save_current_response()
        go_to_image(jump_to - 1)
        st.rerun()

with col4:
    st.write(f"**{current_idx + 1} / {total_images}**")

with col5:
    if current_image in st.session_state.responses:
        st.success("✅ Reviewed")
    else:
        st.warning("⏳ Pending")

st.markdown('</div>', unsafe_allow_html=True)

# Display two versions of images side by side
st.subheader("📸 Image Comparison")
col_img1, col_img2 = st.columns([1, 1])

with col_img1:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("**Original**")
    image_path = os.path.join(IMAGE_DIR, current_image)

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=current_image, use_container_width=True)
    else:
        st.error(f"❌ Image not found: {image_path}")

    st.markdown('</div>', unsafe_allow_html=True)

with col_img2:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("**Checked**")
    image_path_2 = os.path.join(IMAGE_DIR_2, current_image)

    if os.path.exists(image_path_2):
        image_2 = Image.open(image_path_2)
        st.image(image_2, caption=current_image, use_container_width=True)
    else:
        st.warning(f"⚠️ Version 2 not found: {image_path_2}")

    st.markdown('</div>', unsafe_allow_html=True)

# Questions section below images
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

    # Q1: Is objectness appropriate?
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    objectness = st.radio(
        "**1. Is the objectness appropriate?**",
        options=["Yes", "No"],
        index=None,
        key="q1_objectness",
        horizontal=True
    )
    st.session_state.objectness_appropriate = objectness
    st.markdown('</div>', unsafe_allow_html=True)

    # Q2: Objects to add
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.write("**2. If not, should we add objects?**")
    objects_add = st.text_area(
        "List objects to add (one per line or comma-separated):",
        value=saved['objects_to_add'],
        key="q2_add",
        height=100,
        placeholder="e.g., person, car, tree"
    )
    st.session_state.objects_to_add = objects_add
    st.markdown('</div>', unsafe_allow_html=True)

    # Q3: Objects to subtract
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.write("**3. If not, should we subtract objects?**")
    objects_subtract = st.text_area(
        "List objects to remove (one per line or comma-separated):",
        value=saved['objects_to_subtract'],
        key="q3_subtract",
        height=100,
        placeholder="e.g., background, noise"
    )
    st.session_state.objects_to_subtract = objects_subtract
    st.markdown('</div>', unsafe_allow_html=True)

    # Q4: Other comments
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.write("**4. Any other comments?**")
    other = st.text_area(
        "Additional feedback:",
        value=saved['other_comments'],
        key="q4_other",
        height=100,
        placeholder="Any other observations or suggestions..."
    )
    st.session_state.other_comments = other
    st.markdown('</div>', unsafe_allow_html=True)

    # Save button
    if st.button("💾 Save Response", type="primary", use_container_width=True):
        save_current_response()
        st.success("✅ Response saved!")
        st.rerun()