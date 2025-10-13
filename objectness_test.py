import streamlit as st
from PIL import Image
import os

# Image directory
IMAGE_DIR = "./assets"

# Quiz Data with Images
QUESTIONS = [
    {
        "id": 1,
        "image": "ILSVRC2012_val_00028823.JPEG",
        "questions": [
            {"q": "Q1. How many objects do you need to annotate?", "type": "number"},
            {"q": "Q2. Write down all the object classes you found.", "type": "text"},
            {"q": "Q3. Is the classes found in Q2 in the COCO taxonomy?", "type": "radio", "options": ["Yes", "Yes, partially","No"]}
        ],
        # 정답을 여기에 추가 (나중에 수정 가능)
        # 여러 답안 인정: 리스트로 작성
        "answers": {
            "Q1": [5],  # 숫자는 리스트로
            "Q2": ["lemon"],  # 여러 표현 가능
            "Q3": ["No"]  # 대소문자 모두 허용
        }
    },
    {
        "id": 2,
        "image": "ILSVRC2012_val_00002766.JPEG",
        "questions": [
            {"q": "Q4. How many objects do you need to annotate?", "type": "number"},
            {"q": "Q5. Write down all the object classes you found.", "type": "text"},
            {"q": "Q6. Is the classes found in Q5 in the COCO taxonomy?", "type": "radio", "options": ["Yes", "Yes, partially","No"]}
        ],
        "answers": {
            "Q4": [5],
            "Q5": ["Broccoli, Bell Peppers, Garlic"],
            "Q6": ["No"]
        }
    },
    {
        "id": 3,
        "image": "ILSVRC2012_val_00014544.JPEG",
        "questions": [
            {"q": "Q7. How many objects do you need to annotate?", "type": "number"},
            {"q": "Q8. Write down all the object classes you found.", "type": "text"},
            {"q": "Q9. Is the classes found in Q8 in the COCO taxonomy?", "type": "radio", "options": ["Yes", "Yes, partially","No"]}
        ],
        "answers": {
            "Q7": [12],
            "Q8": ["spoon"],
            "Q9": ["Yes",]
        }
    },
    {
        "id": 4,
        "image": "ILSVRC2012_val_00048042.JPEG",
        "questions": [
            {"q": "Q10. How many objects do you need to annotate?", "type": "number"},
            {"q": "Q11. Write down all the object classes you found.", "type": "text"},
            {"q": "Q12. Is the classes found in Q11 in the COCO taxonomy?", "type": "radio", "options": ["Yes", "Yes, partially","No"]}
        ],
        "answers": {
            "Q10": [3],
            "Q11": ["Person, Sports ball"],
            "Q12": ["Yes"]
        }
    }
]

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
    # 현재 질문의 답변 저장
    q_set = QUESTIONS[st.session_state.current_question]
    st.session_state.user_answers[q_set['id']] = st.session_state.current_answers.copy()
    
    # 다음 질문으로
    if st.session_state.current_question < len(QUESTIONS) - 1:
        st.session_state.current_question += 1
        st.session_state.current_answers = {}
    else:
        st.session_state.test_completed = True

def previous_question():
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1
        # 이전 답변 불러오기
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

def normalize_text(text):
    """텍스트 정규화 (공백, 대소문자, 쉼표 처리)"""
    return sorted([s.strip().lower() for s in text.split(',')])

def check_answer(user_answer, correct_answers, q_type):
    """답변 체크 - 여러 정답 허용"""
    # correct_answers는 리스트로 가정
    if not isinstance(correct_answers, list):
        correct_answers = [correct_answers]
    
    if q_type == "number":
        try:
            user_num = int(user_answer)
            return any(user_num == int(ans) for ans in correct_answers)
        except:
            return False
            
    elif q_type == "text":
        # 사용자 답변 정규화
        user_text = str(user_answer).strip().lower()
        user_classes = normalize_text(user_text)
        
        # 여러 정답 중 하나라도 맞으면 OK
        for correct in correct_answers:
            correct_text = str(correct).strip().lower()
            correct_classes = normalize_text(correct_text)
            if user_classes == correct_classes:
                return True
        return False
        
    elif q_type == "radio":
        user_text = str(user_answer).strip().lower()
        return any(user_text == str(ans).strip().lower() for ans in correct_answers)
    
    return False

# Main App
st.set_page_config(page_title="COCO Image Annotation Test", page_icon="🖼️", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# Welcome Screen
if not st.session_state.test_started:
    st.markdown("""
    <div class="main-header">
        <h1>🖼️ COCO Image Annotation Test</h1>
        <p style="font-size: 18px;">Identify objects in images and verify COCO taxonomy knowledge</p>
        <p style="font-size: 16px; opacity: 0.9;">4 Image Sets | Multiple Questions per Image</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ### 📋 Instructions
    - Look at each image carefully
    - Count how many objects need to be annotated
    - List all object classes you can identify
    - Verify if the classes are in COCO taxonomy
    - Click "Submit Answers" to proceed to the next image
    """)
    
    st.warning("""
    ### 📁 Image Files Required
    Make sure the following images are in the `./assets/` directory:
    - assets/ILSVRC2012_val_00028823.JPEG
    - assets/ILSVRC2012_val_00002766.JPEG
    - assets/ILSVRC2012_val_00014544.JPEG
    - assets/ILSVRC2012_val_00048042.JPEG
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
    st.write(f"**Image {st.session_state.current_question + 1} / {len(QUESTIONS)}**")
    
    # Image and Questions in columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        
        # 이미지 로드 (assets 디렉토리에서)
        image_path = os.path.join(IMAGE_DIR, q_set['image'])
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=q_set['image'], width=400)
        else:
            st.error(f"❌ Image not found: {image_path}")
            st.info("Please upload the image file to the `./assets/` directory.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Questions")
        
        # 각 질문에 대한 입력
        for i, question in enumerate(q_set['questions']):
            q_key = f"q{q_set['id']}_{i}"
            
            st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
            st.write(f"**{question['q']}**")
            
            if question['type'] == "number":
                answer = st.number_input(
                    "Your answer:",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.current_answers.get(q_key, 0),
                    key=f"input_{q_key}"
                )
                st.session_state.current_answers[q_key] = answer
                
            elif question['type'] == "text":
                answer = st.text_input(
                    "Your answer (separate multiple items with commas):",
                    value=st.session_state.current_answers.get(q_key, ""),
                    key=f"input_{q_key}",
                    placeholder="e.g., dog, cat, person"
                )
                st.session_state.current_answers[q_key] = answer
                
            elif question['type'] == "radio":
                answer = st.radio(
                    "Your answer:",
                    question['options'],
                    index=question['options'].index(st.session_state.current_answers.get(q_key, question['options'][0])) if q_key in st.session_state.current_answers else 0,
                    key=f"input_{q_key}"
                )
                st.session_state.current_answers[q_key] = answer
            
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
    correct_answers = 0
    
    # 각 이미지별 결과 표시
    for q_set in QUESTIONS:
        st.subheader(f"Image {q_set['id']}: {q_set['image']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image_path = os.path.join(IMAGE_DIR, q_set['image'])
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
        
        with col2:
            user_ans = st.session_state.user_answers.get(q_set['id'], {})
            
            for i, question in enumerate(q_set['questions']):
                q_key = f"q{q_set['id']}_{i}"
                user_answer = user_ans.get(q_key, "No answer")
                
                # 정답 가져오기 (Q1, Q4, Q7, Q10 형식)
                q_number = question['q'].split('.')[0].strip()
                correct_answers = q_set['answers'].get(q_number, ["Not set"])
                
                is_correct = check_answer(user_answer, correct_answers, question['type'])
                total_questions += 1
                if is_correct:
                    correct_answers += 1
                
                # 정답 표시용 (첫 번째 정답만 표시)
                display_answer = correct_answers[0] if isinstance(correct_answers, list) else correct_answers
                
                if is_correct:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>{question['q']}</strong><br>
                        ✅ Your answer: <strong>{user_answer}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>{question['q']}</strong><br>
                        ❌ Your answer: <strong>{user_answer}</strong><br>
                        ✓ Correct answer: <strong>{display_answer}</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
    
    # Final Score
    percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    st.markdown(f"""
    <div class="main-header">
        <h2>Final Score: {correct_answers}/{total_questions}</h2>
        <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Restart button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Restart Test", on_click=restart_test, type="primary", use_container_width=True)