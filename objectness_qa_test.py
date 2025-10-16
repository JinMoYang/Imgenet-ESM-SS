import streamlit as st
from PIL import Image
import os
import glob
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# Configuration
IMAGE_DIR = "./assets"
WORKSHEET_NAME = "Reviews"

@st.cache_resource
def get_gsheet_connection():
    """Initialize Google Sheets connection with Service Account"""
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
        
        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=100, cols=10)
            # Add headers
            worksheet.append_row([
                'Timestamp', 'Reviewer', 'Image Name',
                'Objectness Appropriate', 'Objects to Add',
                'Objects to Subtract', 'Other Comments'
            ])
        
        return worksheet
        
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        st.info("Make sure you've set up Service Account credentials in secrets")
        return None

def save_to_google_sheets(reviewer_name, responses):
    """Save all responses to Google Sheets"""
    if not reviewer_name:
        st.error("Please enter your name before submitting!")
        return False, 0
    
    if not responses:
        st.warning("No reviews to submit yet!")
        return False, 0
    
    worksheet = get_gsheet_connection()
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
        st.session_state.objectness_appropriate = response.get('objectness_appropriate', 'Yes')
        st.session_state.objects_to_add = response.get('objects_to_add', '')
        st.session_state.objects_to_subtract = response.get('objects_to_subtract', '')
        st.session_state.other_comments = response.get('other_comments', '')
    else:
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
    ### 1. Google Sheets Setup with Service Account
    
    **One-time setup required for automatic Google Sheets integration:**
    
    #### A. Create Service Account (5 minutes)
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project (or select existing)
    3. Enable APIs:
       - **Google Sheets API**
       - **Google Drive API**
    4. Create Service Account:
       - Go to **APIs & Services → Credentials**
       - Click **Create Credentials → Service Account**
       - Name it (e.g., "streamlit-app")
       - Click **Done**
    5. Generate Key:
       - Click on the created service account
       - Go to **Keys** tab
       - Click **Add Key → Create New Key → JSON**
       - Download the JSON file
    
    #### B. Share Your Google Sheet
    1. Open your Google Sheet
    2. Click **Share** button
    3. Copy the `client_email` from the JSON file (looks like: `something@project.iam.gserviceaccount.com`)
    4. Add this email as **Editor**
    
    #### C. Add to Streamlit Secrets
    
    **For Local Development:**
    
    Create `.streamlit/secrets.toml`:
    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "your-project-id"
    private_key_id = "key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\\nYOUR_KEY\\n-----END PRIVATE KEY-----\\n"
    client_email = "your-service-account@project.iam.gserviceaccount.com"
    client_id = "123456789"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-email"
    
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
    ```
    
    **For Streamlit Cloud:**
    - Go to App Settings → Secrets
    - Paste the same content
    
    #### D. Requirements
    Make sure `requirements.txt` includes:
    ```
    gspread>=6.0.0
    google-auth>=2.0.0
    ```
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
    - All reviews automatically saved to your Google Sheet
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