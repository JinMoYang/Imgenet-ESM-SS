import streamlit as st
import time

# Quiz Data
QUESTIONS = [
    {
        "id": 1,
        "definition": "A human being regarded as an individual.",
        "options": ["child", "person", "woman", "human", "man"],
        "answer": "person",
        "category": "Person"
    },
    {
        "id": 2,
        "definition": "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, non-retractable claws, and a barking, howling, or whining voice.",
        "options": ["cat", "sheep", "dog", "horse", "cow"],
        "answer": "dog",
        "category": "Animal"
    },
    {
        "id": 3,
        "definition": "A mechanical or electrical device for measuring time, indicating hours, minutes, and sometimes seconds by hands on a round dial or by displayed figures.",
        "options": ["tv", "keyboard", "remote", "clock", "cell phone"],
        "answer": "clock",
        "category": "Indoor"
    },
    {
        "id": 4,
        "definition": "A dish of Italian origin consisting of a flat, round base of dough baked with a topping of tomato sauce and cheese, typically with added meat or vegetables.",
        "options": ["donut", "sandwich", "hot dog", "pizza", "cake"],
        "answer": "pizza",
        "category": "Kitchen"
    },
    {
        "id": 5,
        "definition": "A road vehicle, typically with four wheels, powered by an internal combustion engine or electric motor and able to carry a small number of people.",
        "options": ["bicycle", "motorcycle", "car", "bus", "truck"],
        "answer": "car",
        "category": "Vehicle"
    },
    {
        "id": 6,
        "definition": "A piece of furniture for sleep or rest, typically a framework with a mattress and coverings.",
        "options": ["chair", "bench", "couch", "bed", "dining table"],
        "answer": "bed",
        "category": "Furniture"
    },
    {
        "id": 7,
        "definition": "The round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh.",
        "options": ["carrot", "apple", "banana", "orange", "broccoli"],
        "answer": "apple",
        "category": "Kitchen"
    },
    {
        "id": 8,
        "definition": "A separate seat for one person, typically with a back and four legs.",
        "options": ["bed", "dining table", "chair", "couch", "bench"],
        "answer": "chair",
        "category": "Furniture"
    },
    {
        "id": 9,
        "definition": "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.",
        "options": ["bear", "dog", "sheep", "cat", "horse"],
        "answer": "cat",
        "category": "Animal"
    },
    {
        "id": 10,
        "definition": "A device with a screen for receiving television signals and displaying images and sound.",
        "options": ["microwave", "clock", "cell phone", "laptop", "tv"],
        "answer": "tv",
        "category": "Electronic"
    },
    {
        "id": 11,
        "definition": "A round juicy citrus fruit with a tough bright reddish-yellow rind.",
        "options": ["broccoli", "banana", "orange", "apple", "carrot"],
        "answer": "orange",
        "category": "Kitchen"
    },
    {
        "id": 12,
        "definition": "A red octagonal road sign bearing the word \"STOP\" in white letters, indicating that vehicles must come to a complete halt.",
        "options": ["parking meter", "bench", "fire hydrant", "stop sign", "traffic light"],
        "answer": "stop sign",
        "category": "Outdoor"
    },
    {
        "id": 13,
        "definition": "A vegetable with a green stem or stems and a compact head of small green or purplish flower buds, eaten as a vegetable.",
        "options": ["orange", "apple", "broccoli", "banana", "carrot"],
        "answer": "broccoli",
        "category": "Kitchen"
    },
    {
        "id": 14,
        "definition": "A bag with shoulder straps that allow it to be carried on one's back.",
        "options": ["umbrella", "tie", "backpack", "suitcase", "handbag"],
        "answer": "backpack",
        "category": "Accessory"
    },
    {
        "id": 15,
        "definition": "An appliance or compartment that is artificially kept cool and used to store food and drink.",
        "options": ["toaster", "oven", "microwave", "refrigerator", "sink"],
        "answer": "refrigerator",
        "category": "Appliance"
    },
    {
        "id": 16,
        "definition": "A ball used in various athletic games and sports.",
        "options": ["kite", "tennis racket", "frisbee", "sports ball", "baseball bat"],
        "answer": "sports ball",
        "category": "Sports"
    },
    {
        "id": 17,
        "definition": "A vehicle with two wheels held in a frame one behind the other, propelled by pedals and steered with handlebars attached to the front wheel.",
        "options": ["bus", "skateboard", "motorcycle", "bicycle", "car"],
        "answer": "bicycle",
        "category": "Vehicle"
    },
    {
        "id": 18,
        "definition": "A set of automatically operated colored lights, typically red, amber, and green, for controlling traffic at road junctions and pedestrian crossings.",
        "options": ["fire hydrant", "parking meter", "bench", "traffic light", "stop sign"],
        "answer": "traffic light",
        "category": "Outdoor"
    },
    {
        "id": 19,
        "definition": "A long seat for several people, typically made of wood or stone.",
        "options": ["couch", "dining table", "bench", "chair", "bed"],
        "answer": "bench",
        "category": "Outdoor"
    },
    {
        "id": 20,
        "definition": "A large bowl for urinating or defecating into, typically plumbed into a sewage system and with a flushing device.",
        "options": ["chair", "sink", "bed", "potted plant", "toilet"],
        "answer": "toilet",
        "category": "Furniture"
    },
    {
        "id": 21,
        "definition": "A small vessel propelled on water by oars, sails, or an engine.",
        "options": ["train", "airplane", "bus", "boat", "car"],
        "answer": "boat",
        "category": "Vehicle"
    },
    {
        "id": 22,
        "definition": "An African wild horse with black-and-white stripes and an erect mane.",
        "options": ["giraffe", "cow", "zebra", "horse", "elephant"],
        "answer": "zebra",
        "category": "Animal"
    },
    {
        "id": 23,
        "definition": "A water outlet with a nozzle to which a fire hose can be attached, typically placed at regular intervals along streets.",
        "options": ["stop sign", "parking meter", "fire hydrant", "traffic light", "bench"],
        "answer": "fire hydrant",
        "category": "Outdoor"
    },
    {
        "id": 24,
        "definition": "An instrument used for cutting cloth, paper, and other material, consisting of two blades laid one on top of the other and fastened in the middle so as to allow them to be opened and closed by a thumb and finger inserted through rings on the end of their handles.",
        "options": ["toothbrush", "scissors", "knife", "fork", "spoon"],
        "answer": "scissors",
        "category": "Indoor"
    },
    {
        "id": 25,
        "definition": "A toy consisting of a light frame with thin material stretched over it, flown in the wind at the end of a long string.",
        "options": ["skateboard", "surfboard", "sports ball", "frisbee", "kite"],
        "answer": "kite",
        "category": "Sports"
    },
    {
        "id": 26,
        "definition": "A coin-operated device beside a parking space that registers the time purchased for parking and indicates when this has expired.",
        "options": ["traffic light", "bench", "parking meter", "stop sign", "fire hydrant"],
        "answer": "parking meter",
        "category": "Outdoor"
    },
    {
        "id": 27,
        "definition": "Each of a pair of long, narrow pieces of hard flexible material, typically pointed and turned up at the front, fastened under the feet for gliding over snow.",
        "options": ["surfboard", "frisbee", "snowboard", "skateboard", "skis"],
        "answer": "skis",
        "category": "Sports"
    },
    {
        "id": 28,
        "definition": "A racket with a round or oval frame strung with catgut, nylon, or similar material, used for playing tennis.",
        "options": ["baseball glove", "baseball bat", "frisbee", "sports ball", "tennis racket"],
        "answer": "tennis racket",
        "category": "Sports"
    },
    {
        "id": 29,
        "definition": "An electrical device for drying a person's hair by blowing warm air over it.",
        "options": ["remote", "clock", "mouse", "toothbrush", "hair drier"],
        "answer": "hair drier",
        "category": "Indoor"
    },
    {
        "id": 30,
        "definition": "A decorative container, typically made of glass or china and used as an ornament or for displaying cut flowers.",
        "options": ["bowl", "vase", "cup", "potted plant", "bottle"],
        "answer": "vase",
        "category": "Indoor"
    }
]

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.user_answers = []
    st.session_state.score = 0
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.selected_answer = None
    st.session_state.show_feedback = False

def start_test():
    st.session_state.test_started = True
    st.session_state.current_question = 0
    st.session_state.user_answers = []
    st.session_state.score = 0
    st.session_state.test_completed = False
    st.session_state.selected_answer = None
    st.session_state.show_feedback = False

def submit_answer():
    if st.session_state.selected_answer is None:
        st.warning("⚠️ Please select an answer before submitting!")
        return
    
    q = QUESTIONS[st.session_state.current_question]
    user_answer = st.session_state.selected_answer
    is_correct = user_answer == q['answer']
    
    st.session_state.user_answers.append({
        'question_id': q['id'],
        'user_answer': user_answer,
        'correct_answer': q['answer'],
        'is_correct': is_correct
    })
    
    if is_correct:
        st.session_state.score += 1
    
    st.session_state.show_feedback = True

def next_question():
    st.session_state.current_question += 1
    st.session_state.selected_answer = None
    st.session_state.show_feedback = False
    
    if st.session_state.current_question >= len(QUESTIONS):
        st.session_state.test_completed = True

def restart_test():
    st.session_state.current_question = 0
    st.session_state.user_answers = []
    st.session_state.score = 0
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.selected_answer = None
    st.session_state.show_feedback = False

# Main App
st.set_page_config(page_title="COCO Label Understanding Test", page_icon="📝", layout="wide")

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
        <h1>📝 COCO Label Understanding Test</h1>
        <p style="font-size: 18px;">Test your knowledge of COCO dataset categories</p>
        <p style="font-size: 16px; opacity: 0.9;">30 Questions | 90% Required to Pass</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ### 📋 Instructions
    - Read each definition carefully
    - Select the correct COCO category from the options
    - Click "Submit Answer" to check your answer
    - You need to score 90% (27/30) to pass this test
    - You will see immediate feedback after each answer
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Start Test", on_click=start_test, type="primary", use_container_width=True)

# Test in Progress
elif st.session_state.test_started and not st.session_state.test_completed:
    q = QUESTIONS[st.session_state.current_question]
    
    # Progress bar
    progress = st.session_state.current_question / len(QUESTIONS)
    st.progress(progress)
    st.write(f"**Question {st.session_state.current_question + 1} / {len(QUESTIONS)}**")
    
    # Question
    st.markdown(f"""
    <div class="question-box">
        <h2>Question {q['id']}</h2>
        <p style="font-size: 18px; line-height: 1.6;">{q['definition']}</p>
        <p style="margin-top: 15px; font-size: 14px; opacity: 0.9;">Category: {q['category']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    selected = st.radio(
        "Select your answer:",
        q['options'],
        key=f"q_{st.session_state.current_question}",
        index=None
    )
    
    if selected:
        st.session_state.selected_answer = selected
    
    # Feedback
    if st.session_state.show_feedback:
        is_correct = st.session_state.user_answers[-1]['is_correct']
        if is_correct:
            st.markdown("""
            <div class="success-box">
                <strong style="color: #155724;">✅ Correct!</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <strong style="color: #721c24;">❌ Incorrect!</strong><br>
                <span style="color: #721c24;">The correct answer is: <strong>{q['answer']}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        
        st.button("Next Question →", on_click=next_question, type="primary")
    else:
        st.button("Submit Answer", on_click=submit_answer, type="primary")

# Results Screen
elif st.session_state.test_completed:
    percentage = (st.session_state.score / len(QUESTIONS)) * 100
    passed = percentage >= 90
    
    # Result summary
    if passed:
        st.markdown(f"""
        <div class="result-box">
            <h1 style="font-size: 48px; margin-bottom: 20px;">🎉 PASSED!</h1>
            <h2>Your Score: {st.session_state.score}/{len(QUESTIONS)}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                Congratulations! You may proceed to objectness test.
            <a href="https://imgenet-esm-ss-h52gbsgcpnzn7jxsnqi3x9.streamlit.app/" target="_blank" style="color: #155724; text-decoration: underline;">Click here to proceed to objectness test</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="failed-box">
            <h1 style="font-size: 48px; margin-bottom: 20px;">❌ FAILED</h1>
            <h2>Your Score: {st.session_state.score}/{len(QUESTIONS)}</h2>
            <p style="font-size: 24px;">Percentage: {percentage:.1f}%</p>
            <p style="margin-top: 20px; font-size: 16px; opacity: 0.9;">
                Please review the COCO categories and try again. You need 90% to pass.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    st.subheader("📊 Detailed Results")
    
    results_data = []
    for ans in st.session_state.user_answers:
        results_data.append({
            "Question": ans['question_id'],
            "Your Answer": ans['user_answer'],
            "Correct Answer": ans['correct_answer'],
            "Result": "✅" if ans['is_correct'] else "❌"
        })
    
    st.dataframe(results_data, use_container_width=True)
    
    # Restart button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Restart Test", on_click=restart_test, type="primary", use_container_width=True)