import streamlit as st
from PIL import Image
import os
import glob
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# Configuration
IMAGE_DIR = "./assets"
WORKSHEET_NAME = "Reviews"

@st.cache_resource
def get_gsheet_connection():
    """Initialize Google Sheets connection"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        return conn
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        st.info("Make sure you've added the spreadsheet URL to secrets.toml")
        return None

def save_to_google_sheets(reviewer_name, responses):
    """Save all responses to Google Sheets"""
    if not reviewer_name:
        st.error("Please enter your name before submitting!")
        return False, 0
    
    if not responses:
        st.warning("No reviews to submit yet!")
        return False, 0
    
    conn = get_gsheet_connection()
    if conn is None:
        return False, 0
    
    try:
        # Try to read existing data
        try:
            existing_df = conn.read(worksheet=WORKSHEET_NAME, ttl=0)
            # Ensure columns exist
            required_columns = [
                'Timestamp', 'Reviewer', 'Image Name',
                'Objectness Appropriate', 'Objects to Add',
                'Objects to Subtract', 'Other Comments'
            ]
            for col in required_columns:
                if col not in existing_df.columns:
                    existing_df[col] = ''
        except:
            # Create new dataframe if sheet is empty
            existing_df = pd.DataFrame(columns=[
                'Timestamp', 'Reviewer', 'Image Name',
                'Objectness Appropriate', 'Objects to Add',
                'Objects to Subtract', 'Other Comments'
            ])
        
        # Prepare new rows
        new_rows = []
        for image_name, response in responses.items():
            new_rows.append({
                'Timestamp': response['timestamp'],
                'Reviewer': reviewer_name,
                'Image Name': image_name,
                'Objectness Appropriate': response['objectness_appropriate'],
                'Objects to Add': response['objects_to_add'],
                'Objects to Subtract': response['objects_to_subtract'],
                'Other Comments': response['other_comments']
            })
        
        new_df = pd.DataFrame(new_rows)
        
        # Combine and update
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Write to sheet
        conn.update(worksheet=WORKSHEET_NAME, data=updated_df)
        
        return True, len(new_rows)
        
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
    ### 1. Google Sheets Setup (Simple Method)

    **No Google Cloud Account Required!**

    **Steps:**

    1. **Create a Google Spreadsheet**
       - Go to [Google Sheets](https://sheets.google.com)
       - Create a new spreadsheet
       - Name it anything you want (e.g., "Objectness QA Reviews")
       - Create a worksheet tab named: **"Reviews"**

    2. **Share the Spreadsheet**
       - Click the "Share" button (top right)
       - Change to: **"Anyone with the link"**
       - Set permission to: **"Editor"**
       - Copy the spreadsheet URL

    3. **Add to Streamlit Secrets**
       - Create `.streamlit/secrets.toml` file in your project directory
       - Add the following:
       ```toml
       [connections.gsheets]
       spreadsheet = "YOUR_GOOGLE_SHEET_URL_HERE"
       ```
       
    4. **For Streamlit Cloud**
       - Go to App Settings → Secrets
       - Paste the same content:
       ```
       [connections.gsheets]
       spreadsheet = "YOUR_GOOGLE_SHEET_URL_HERE"
       ```

    **That's it!** Much simpler than Service Account setup.
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
                use_container_width=True
            )
        else:
            st.button(
                "🚀 Start Review",
                type="primary",
                use_container_width=True,
                disabled=True
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
        with st.spinner("Submitting to Google Sheets..."):
            success, count = save_to_google_sheets(
                st.session_state.reviewer_name,
                st.session_state.responses
            )
            if success:
                st.success(f"✅ Successfully submitted {count} reviews!")
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