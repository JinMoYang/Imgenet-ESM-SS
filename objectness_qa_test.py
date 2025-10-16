import streamlit as st
from PIL import Image
import os
import glob
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# Configuration
IMAGE_DIR = "./assets"
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Google Sheets Configuration
SPREADSHEET_NAME = "Objectness QA Test Results"  # Update with your spreadsheet name
SHEET_NAME = "Reviews"  # Tab name for reviews

def init_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        # Load credentials from streamlit secrets or local file
        if 'gcp_service_account' in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        else:
            # Fallback to local credentials file
            credentials = Credentials.from_service_account_file(
                'credentials.json',
                scopes=SCOPES
            )

        client = gspread.authorize(credentials)
        spreadsheet = client.open(SPREADSHEET_NAME)
        sheet = spreadsheet.worksheet(SHEET_NAME)
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        return None

def save_to_google_sheets(sheet, data):
    """Save review data to Google Sheets"""
    try:
        # Append row to sheet
        sheet.append_row(data)
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {str(e)}")
        return False

def get_image_list():
    """Get list of all JPEG images in the assets directory"""
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.JPEG"))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.JPG")))
    # Sort by filename
    image_files.sort()
    # Return just filenames
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
        st.session_state.objectness_appropriate = response.get('objectness_appropriate', 'Yes')
        st.session_state.objects_to_add = response.get('objects_to_add', '')
        st.session_state.objects_to_subtract = response.get('objects_to_subtract', '')
        st.session_state.other_comments = response.get('other_comments', '')
    else:
        # Reset to defaults
        st.session_state.objectness_appropriate = 'Yes'
        st.session_state.objects_to_add = ''
        st.session_state.objects_to_subtract = ''
        st.session_state.other_comments = ''

def save_current_response():
    """Save current response to session state"""
    current_image = st.session_state.images[st.session_state.current_image_idx]
    st.session_state.responses[current_image] = {
        'objectness_appropriate': st.session_state.get('objectness_appropriate', 'Yes'),
        'objects_to_add': st.session_state.get('objects_to_add', ''),
        'objects_to_subtract': st.session_state.get('objects_to_subtract', ''),
        'other_comments': st.session_state.get('other_comments', ''),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def submit_to_sheets():
    """Submit all responses to Google Sheets"""
    if not st.session_state.reviewer_name:
        st.error("Please enter your name before submitting!")
        return False

    sheet = init_google_sheets()
    if sheet is None:
        return False

    # Check if headers exist, if not create them
    try:
        headers = sheet.row_values(1)
        if not headers:
            sheet.append_row([
                'Timestamp', 'Reviewer', 'Image Name',
                'Objectness Appropriate', 'Objects to Add',
                'Objects to Subtract', 'Other Comments'
            ])
    except:
        sheet.append_row([
            'Timestamp', 'Reviewer', 'Image Name',
            'Objectness Appropriate', 'Objects to Add',
            'Objects to Subtract', 'Other Comments'
        ])

    # Submit all responses
    success_count = 0
    for image_name, response in st.session_state.responses.items():
        data = [
            response['timestamp'],
            st.session_state.reviewer_name,
            image_name,
            response['objectness_appropriate'],
            response['objects_to_add'],
            response['objects_to_subtract'],
            response['other_comments']
        ]
        if save_to_google_sheets(sheet, data):
            success_count += 1

    if success_count > 0:
        st.success(f"Successfully submitted {success_count} reviews to Google Sheets!")
        return True
    return False

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
    .stButton>button {
        width: 100%;
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
    ### 1. Google Sheets Setup

    This application requires Google Sheets API access to save your reviews.

    **Steps to set up:**

    1. **Create a Google Cloud Project**
       - Go to [Google Cloud Console](https://console.cloud.google.com/)
       - Create a new project
       - Enable **Google Sheets API** and **Google Drive API**

    2. **Create Service Account**
       - Go to IAM & Admin > Service Accounts
       - Create a service account
       - Download the JSON credentials file
       - Save it as `credentials.json` in the Tests directory

    3. **Create Google Spreadsheet**
       - Create a new spreadsheet named: **"Objectness QA Test Results"**
       - Create a worksheet tab named: **"Reviews"**
       - Share the spreadsheet with the service account email (found in credentials.json)
       - Give it Editor permissions

    4. **For Streamlit Cloud Deployment**
       - Add credentials to Streamlit Secrets instead of using credentials.json
       - See [Streamlit Secrets Documentation](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
    """)

    st.markdown("---")

    st.markdown("""
    ### 2. How to Use This App

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
    - All your reviews will be saved to the spreadsheet
    - Each row includes: timestamp, your name, image name, and all responses
    """)

    st.markdown("---")

    st.markdown("""
    ### 3. Data Format

    Your reviews will be saved to Google Sheets with the following columns:

    | Column | Description |
    |--------|-------------|
    | Timestamp | When the review was submitted |
    | Reviewer | Your name |
    | Image Name | Filename of the reviewed image |
    | Objectness Appropriate | Yes/No response |
    | Objects to Add | List of objects to add |
    | Objects to Subtract | List of objects to remove |
    | Other Comments | Additional feedback |
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

    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if total_images > 0:
            st.button(
                "🚀 Start Review",
                on_click=start_app,
                type="primary",
                use_container_width=True,
                key="start_button"
            )
        else:
            st.button(
                "🚀 Start Review",
                type="primary",
                use_container_width=True,
                disabled=True,
                key="start_button_disabled"
            )

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

    # Submit all button
    st.header("Submit Reviews")
    if st.button("📤 Submit All to Google Sheets", type="primary", use_container_width=True):
        if reviewed_count == 0:
            st.warning("No reviews to submit yet!")
        else:
            with st.spinner("Submitting to Google Sheets..."):
                if submit_to_sheets():
                    st.balloons()

    st.divider()

    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        1. **Enter your name** in the sidebar
        2. **Review each image** and answer the questions
        3. **Navigate** using Prev/Next buttons or jump to a specific image
        4. Your responses are **automatically saved** as you type
        5. Click **Submit All** when you've reviewed all images

        **Questions:**
        - **Objectness Appropriate?** - Does the annotation correctly identify all objects?
        - **Objects to Add** - List any objects that should be annotated but aren't
        - **Objects to Subtract** - List any objects that shouldn't be annotated
        - **Other Comments** - Any additional feedback
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
    # Jump to image number
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
    # Show if this image has been reviewed
    if current_image in st.session_state.responses:
        st.success("✅ Reviewed")
    else:
        st.warning("⏳ Pending")

st.markdown('</div>', unsafe_allow_html=True)

# Image and Questions
col_img, col_qa = st.columns([1, 1])

with col_img:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    image_path = os.path.join(IMAGE_DIR, current_image)

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=current_image, use_container_width=True)
    else:
        st.error(f"❌ Image not found: {image_path}")

    st.markdown('</div>', unsafe_allow_html=True)

with col_qa:
    st.subheader("Review Questions")

    # Load saved response if exists
    if current_image in st.session_state.responses:
        saved = st.session_state.responses[current_image]
    else:
        saved = {
            'objectness_appropriate': 'Yes',
            'objects_to_add': '',
            'objects_to_subtract': '',
            'other_comments': ''
        }

    # Q1: Is objectness appropriate?
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    objectness = st.radio(
        "**1. Is the objectness appropriate?**",
        options=["Yes", "No"],
        index=0 if saved['objectness_appropriate'] == 'Yes' else 1,
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

# Auto-save on navigation
# The save_current_response() is called before navigation in the button callbacks
