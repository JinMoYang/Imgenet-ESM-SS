import streamlit as st
from PIL import Image
import os

# Image directory
IMAGE_DIR = "./assets/tag_test"

# Quiz Data with Images
# 이미지와 정답을 아래 형식에 맞춰 추가하세요.
# - image: 이미지 파일명 (assets/tag_test/ 폴더에 넣기)
# - answer_object: 태그를 추가해야 할 object 이름 (소문자, COCO 카테고리)
# - answer_tag: 추가해야 할 태그 (ishole, iscrowd, isreflected, isdepicted, isthrough 중 하나)
QUESTIONS = [
    {
        "id": 1,
        "image": "tag_q1.JPEG",
        "hint": "Look at the image carefully and determine which object needs an additional tag.",
        "answer_object": "cat",         
        "answer_tag": "isreflected",   
    },
    {
        "id": 2,
        "image": "tag_q2.JPEG",
        "hint": "Look at the image carefully and determine which object needs an additional tag.",
        "answer_object": "stop sign",
        "answer_tag": "isthrough",
    },
    {
        "id": 3,
        "image": "tag_q3.JPEG",
        "hint": "Look at the image carefully and determine which object needs an additional tag.",
        "answer_object": "person",
        "answer_tag": "isdepicted",
    },
    {
        "id": 4,
        "image": "tag_q4.JPEG",
        "hint": "Look at the image carefully and determine which object needs an additional tag.",
        "answer_object": "chair",
        "answer_tag": "ishole",
    },
    {
        "id": 5,
        "image": "tag_q5.JPEG",
        "hint": "Look at the image carefully and determine which object needs an additional tag.",
        "answer_object": "cow",
        "answer_tag": "iscrowd",
    },
]

TAG_OPTIONS = ["ishole", "iscrowd", "isreflected", "isdepicted", "isthrough"]

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.user_answers = {}
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.current_answers = {}

def start_test():
    st.session_state.test_started = True
    st.session_state.current_question = 0
    st.session_state.user_answers = {}
    st.session_state.test_completed = False
    st.session_state.current_answers = {}

def submit_answers():
    q_set = QUESTIONS[st.session_state.current_question]
    st.session_state.user_answers[q_set['id']] = st.session_state.current_answers.copy()

    if st.session_state.current_question < len(QUESTIONS) - 1:
        st.session_state.current_question += 1
        st.session_state.current_answers = {}
    else:
        st.session_state.test_completed = True

def previous_question():
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1
        q_set = QUESTIONS[st.session_state.current_question]
        if q_set['id'] in st.session_state.user_answers:
            st.session_state.current_answers = st.session_state.user_answers[q_set['id']].copy()
        else:
            st.session_state.current_answers = {}

def restart_test():
    st.session_state.current_question = 0
    st.session_state.user_answers = {}
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.current_answers = {}

def check_object_answer(user_answer, correct_answer):
    return user_answer.strip().lower() == correct_answer.strip().lower()

def check_tag_answer(user_answer, correct_answer):
    return user_answer.strip().lower() == correct_answer.strip().lower()

# Main App
st.set_page_config(page_title="Tag Understanding Test", page_icon="🏷️", layout="wide")

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
    .result-box {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .failed-box {
        background: linear-gradient(135deg, #f44336 0%, #e57373 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Welcome Screen
if not st.session_state.test_started:
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ Tag Understanding Test</h1>
        <p style="font-size: 18px;">Test your understanding of annotation tags</p>
        <p style="font-size: 16px; opacity: 0.9;">5 Questions | 80% Required to Pass</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    ### 📋 Instructions
    - Look at each image carefully
    - **Q1**: Identify which object in the image needs an additional tag (enter in **lowercase**, must be a COCO category)
    - **Q2**: Select the appropriate tag for that object
    - Available tags: `ishole`, `iscrowd`, `isreflected`, `isdepicted`, `isthrough`
    - Click "Submit & Next" to proceed to the next question
    - You need to score **80% (8/10)** to pass this test
    """)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Start Test", on_click=start_test, type="primary", use_container_width=True)

# Test in Progress
elif st.session_state.test_started and not st.session_state.test_completed:
    q_set = QUESTIONS[st.session_state.current_question]

    # Progress
    progress = (st.session_state.current_question + 1) / len(QUESTIONS)
    st.progress(progress)
    st.write(f"**Question {st.session_state.current_question + 1} / {len(QUESTIONS)}**")

    # Image and Questions in columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)

        image_path = os.path.join(IMAGE_DIR, q_set['image'])
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=q_set['image'], use_container_width=True)
        else:
            st.error(f"Image not found: {image_path}")
            st.info("Please upload the image file to the `./assets/tag_test/` directory.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Questions")

        # Q1: Which object needs a tag?
        obj_key = f"q{q_set['id']}_object"
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.write(f"**Q{(q_set['id'] - 1) * 2 + 1}. Which object in this image needs an additional tag?**")
        st.caption("Enter in lowercase (must be a COCO category, e.g., person, car, dog)")
        obj_answer = st.text_input(
            "Your answer:",
            value=st.session_state.current_answers.get(obj_key, ""),
            key=f"input_{obj_key}",
            placeholder="e.g., person"
        )
        st.session_state.current_answers[obj_key] = obj_answer
        st.markdown('</div>', unsafe_allow_html=True)

        # Q2: Which tag should be added?
        tag_key = f"q{q_set['id']}_tag"
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.write(f"**Q{(q_set['id'] - 1) * 2 + 2}. Which tag should be added to this object?**")

        default_index = None
        if tag_key in st.session_state.current_answers:
            try:
                default_index = TAG_OPTIONS.index(st.session_state.current_answers[tag_key])
            except ValueError:
                default_index = None

        tag_answer = st.radio(
            "Select the tag:",
            TAG_OPTIONS,
            index=default_index,
            key=f"input_{tag_key}"
        )
        if tag_answer is not None:
            st.session_state.current_answers[tag_key] = tag_answer
        st.markdown('</div>', unsafe_allow_html=True)

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state.current_question > 0:
            st.button("← Previous", on_click=previous_question, use_container_width=True)

    with col3:
        if st.session_state.current_question < len(QUESTIONS) - 1:
            st.button("Submit & Next →", on_click=submit_answers, type="primary", use_container_width=True)
        else:
            st.button("Submit & Finish", on_click=submit_answers, type="primary", use_container_width=True)

# Results Screen
elif st.session_state.test_completed:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Test Results</h1>
        <p style="font-size: 18px;">Review your answers</p>
    </div>
    """, unsafe_allow_html=True)

    total_questions = 0
    correct_count = 0

    for q_set in QUESTIONS:
        st.subheader(f"Question {q_set['id']}: {q_set['image']}")

        col1, col2 = st.columns([1, 1])

        with col1:
            image_path = os.path.join(IMAGE_DIR, q_set['image'])
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, use_container_width=True)

        with col2:
            user_ans = st.session_state.user_answers.get(q_set['id'], {})

            # Check object answer
            obj_key = f"q{q_set['id']}_object"
            user_obj = user_ans.get(obj_key, "No answer")
            obj_correct = check_object_answer(str(user_obj), q_set['answer_object'])
            total_questions += 1
            if obj_correct:
                correct_count += 1

            q_num_obj = (q_set['id'] - 1) * 2 + 1
            if obj_correct:
                st.markdown(f"""
                <div class="success-box">
                    <strong>Q{q_num_obj}. Which object needs a tag?</strong><br>
                    ✅ Your answer: <strong>{user_obj}</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <strong>Q{q_num_obj}. Which object needs a tag?</strong><br>
                    ❌ Your answer: <strong>{user_obj}</strong><br>
                    ✓ Correct answer: <strong>{q_set['answer_object']}</strong>
                </div>
                """, unsafe_allow_html=True)

            # Check tag answer
            tag_key = f"q{q_set['id']}_tag"
            user_tag = user_ans.get(tag_key, "No answer")
            tag_correct = check_tag_answer(str(user_tag), q_set['answer_tag'])
            total_questions += 1
            if tag_correct:
                correct_count += 1

            q_num_tag = (q_set['id'] - 1) * 2 + 2
            if tag_correct:
                st.markdown(f"""
                <div class="success-box">
                    <strong>Q{q_num_tag}. Which tag should be added?</strong><br>
                    ✅ Your answer: <strong>{user_tag}</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    <strong>Q{q_num_tag}. Which tag should be added?</strong><br>
                    ❌ Your answer: <strong>{user_tag}</strong><br>
                    ✓ Correct answer: <strong>{q_set['answer_tag']}</strong>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

    # Final Score
    percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0

    if percentage >= 80:
        st.markdown(f"""
        <div class="result-box">
            <h1 style="font-size: 48px; margin-bottom: 20px;">🎉 PASSED!</h1>
            <h2>Final Score: {correct_count}/{total_questions}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                Congratulations! You have demonstrated understanding of annotation tags.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="failed-box">
            <h1 style="font-size: 48px; margin-bottom: 20px;">❌ FAILED</h1>
            <h2>Final Score: {correct_count}/{total_questions}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                Please review the tag definitions and try again. You need 80% to pass.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Restart button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Restart Test", on_click=restart_test, type="primary", use_container_width=True)
