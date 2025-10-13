# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("COCO Test Dashboard")

# 노트북 코드를 여기에 복사
# 단, print() → st.write()
# plt.show() → st.pyplot(fig)
"""
Annotator Qualification Test - Part 1: COCO Label Understanding
30 Multiple Choice Questions

Usage with Voila:
1. Save this as 'coco_test.ipynb'
2. Run: voila coco_test.ipynb
3. Code will be hidden, only UI will be visible
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Quiz Data
QUESTIONS = [
    {
        "id": 1,
        "definition": "A human being regarded as an individual.",
        "options": ["person", "man", "woman", "child", "human"],
        "answer": "person",
        "category": "Person"
    },
    {
        "id": 2,
        "definition": "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, non-retractable claws, and a barking, howling, or whining voice.",
        "options": ["dog", "cat", "horse", "cow", "sheep"],
        "answer": "dog",
        "category": "Animal"
    },
    {
        "id": 3,
        "definition": "A mechanical or electrical device for measuring time, indicating hours, minutes, and sometimes seconds by hands on a round dial or by displayed figures.",
        "options": ["clock", "tv", "remote", "cell phone", "keyboard"],
        "answer": "clock",
        "category": "Indoor"
    },
    {
        "id": 4,
        "definition": "A dish of Italian origin consisting of a flat, round base of dough baked with a topping of tomato sauce and cheese, typically with added meat or vegetables.",
        "options": ["pizza", "sandwich", "hot dog", "donut", "cake"],
        "answer": "pizza",
        "category": "Kitchen"
    },
    {
        "id": 5,
        "definition": "A road vehicle, typically with four wheels, powered by an internal combustion engine or electric motor and able to carry a small number of people.",
        "options": ["car", "bus", "truck", "motorcycle", "bicycle"],
        "answer": "car",
        "category": "Vehicle"
    },
    {
        "id": 6,
        "definition": "A piece of furniture for sleep or rest, typically a framework with a mattress and coverings.",
        "options": ["bed", "couch", "chair", "bench", "dining table"],
        "answer": "bed",
        "category": "Furniture"
    },
    {
        "id": 7,
        "definition": "The round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh.",
        "options": ["apple", "orange", "banana", "carrot", "broccoli"],
        "answer": "apple",
        "category": "Kitchen"
    },
    {
        "id": 8,
        "definition": "A separate seat for one person, typically with a back and four legs.",
        "options": ["chair", "couch", "bench", "bed", "dining table"],
        "answer": "chair",
        "category": "Furniture"
    },
    {
        "id": 9,
        "definition": "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.",
        "options": ["cat", "dog", "bear", "horse", "sheep"],
        "answer": "cat",
        "category": "Animal"
    },
    {
        "id": 10,
        "definition": "A device with a screen for receiving television signals and displaying images and sound.",
        "options": ["tv", "laptop", "cell phone", "clock", "microwave"],
        "answer": "tv",
        "category": "Electronic"
    },
    {
        "id": 11,
        "definition": "A round juicy citrus fruit with a tough bright reddish-yellow rind.",
        "options": ["orange", "apple", "banana", "carrot", "broccoli"],
        "answer": "orange",
        "category": "Kitchen"
    },
    {
        "id": 12,
        "definition": "A red octagonal road sign bearing the word \"STOP\" in white letters, indicating that vehicles must come to a complete halt.",
        "options": ["stop sign", "traffic light", "fire hydrant", "parking meter", "bench"],
        "answer": "stop sign",
        "category": "Outdoor"
    },
    {
        "id": 13,
        "definition": "A vegetable with a green stem or stems and a compact head of small green or purplish flower buds, eaten as a vegetable.",
        "options": ["broccoli", "carrot", "apple", "orange", "banana"],
        "answer": "broccoli",
        "category": "Kitchen"
    },
    {
        "id": 14,
        "definition": "A bag with shoulder straps that allow it to be carried on one's back.",
        "options": ["backpack", "handbag", "suitcase", "umbrella", "tie"],
        "answer": "backpack",
        "category": "Accessory"
    },
    {
        "id": 15,
        "definition": "An appliance or compartment that is artificially kept cool and used to store food and drink.",
        "options": ["refrigerator", "microwave", "oven", "toaster", "sink"],
        "answer": "refrigerator",
        "category": "Appliance"
    },
    {
        "id": 16,
        "definition": "A ball used in various athletic games and sports.",
        "options": ["sports ball", "frisbee", "kite", "baseball bat", "tennis racket"],
        "answer": "sports ball",
        "category": "Sports"
    },
    {
        "id": 17,
        "definition": "A vehicle with two wheels held in a frame one behind the other, propelled by pedals and steered with handlebars attached to the front wheel.",
        "options": ["bicycle", "motorcycle", "car", "bus", "skateboard"],
        "answer": "bicycle",
        "category": "Vehicle"
    },
    {
        "id": 18,
        "definition": "A set of automatically operated colored lights, typically red, amber, and green, for controlling traffic at road junctions and pedestrian crossings.",
        "options": ["traffic light", "stop sign", "fire hydrant", "parking meter", "bench"],
        "answer": "traffic light",
        "category": "Outdoor"
    },
    {
        "id": 19,
        "definition": "A long seat for several people, typically made of wood or stone.",
        "options": ["bench", "chair", "couch", "bed", "dining table"],
        "answer": "bench",
        "category": "Outdoor"
    },
    {
        "id": 20,
        "definition": "A large bowl for urinating or defecating into, typically plumbed into a sewage system and with a flushing device.",
        "options": ["toilet", "sink", "bed", "chair", "potted plant"],
        "answer": "toilet",
        "category": "Furniture"
    },
    {
        "id": 21,
        "definition": "A small vessel propelled on water by oars, sails, or an engine.",
        "options": ["boat", "car", "bus", "train", "airplane"],
        "answer": "boat",
        "category": "Vehicle"
    },
    {
        "id": 22,
        "definition": "An African wild horse with black-and-white stripes and an erect mane.",
        "options": ["zebra", "horse", "giraffe", "cow", "elephant"],
        "answer": "zebra",
        "category": "Animal"
    },
    {
        "id": 23,
        "definition": "A water outlet with a nozzle to which a fire hose can be attached, typically placed at regular intervals along streets.",
        "options": ["fire hydrant", "stop sign", "traffic light", "parking meter", "bench"],
        "answer": "fire hydrant",
        "category": "Outdoor"
    },
    {
        "id": 24,
        "definition": "An instrument used for cutting cloth, paper, and other material, consisting of two blades laid one on top of the other and fastened in the middle so as to allow them to be opened and closed by a thumb and finger inserted through rings on the end of their handles.",
        "options": ["scissors", "knife", "fork", "spoon", "toothbrush"],
        "answer": "scissors",
        "category": "Indoor"
    },
    {
        "id": 25,
        "definition": "A toy consisting of a light frame with thin material stretched over it, flown in the wind at the end of a long string.",
        "options": ["kite", "frisbee", "sports ball", "skateboard", "surfboard"],
        "answer": "kite",
        "category": "Sports"
    },
    {
        "id": 26,
        "definition": "A coin-operated device beside a parking space that registers the time purchased for parking and indicates when this has expired.",
        "options": ["parking meter", "stop sign", "traffic light", "fire hydrant", "bench"],
        "answer": "parking meter",
        "category": "Outdoor"
    },
    {
        "id": 27,
        "definition": "Each of a pair of long, narrow pieces of hard flexible material, typically pointed and turned up at the front, fastened under the feet for gliding over snow.",
        "options": ["skis", "snowboard", "surfboard", "skateboard", "frisbee"],
        "answer": "skis",
        "category": "Sports"
    },
    {
        "id": 28,
        "definition": "A racket with a round or oval frame strung with catgut, nylon, or similar material, used for playing tennis.",
        "options": ["tennis racket", "baseball bat", "baseball glove", "frisbee", "sports ball"],
        "answer": "tennis racket",
        "category": "Sports"
    },
    {
        "id": 29,
        "definition": "An electrical device for drying a person's hair by blowing warm air over it.",
        "options": ["hair drier", "toothbrush", "remote", "mouse", "clock"],
        "answer": "hair drier",
        "category": "Indoor"
    },
    {
        "id": 30,
        "definition": "A decorative container, typically made of glass or china and used as an ornament or for displaying cut flowers.",
        "options": ["vase", "bowl", "cup", "bottle", "potted plant"],
        "answer": "vase",
        "category": "Indoor"
    }
]

class COCOQualificationTest:
    def __init__(self):
        self.current_question = 0
        self.user_answers = []
        self.score = 0
        
        # UI Components
        self.output = widgets.Output()
        self.question_output = widgets.Output()
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize UI components"""
        # Progress bar
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=len(QUESTIONS),
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#4CAF50'},
            layout=widgets.Layout(width='100%')
        )
        
        # Question container
        self.question_label = widgets.HTML()
        
        # Radio buttons for options
        self.radio_options = widgets.RadioButtons(
            options=[],
            layout=widgets.Layout(width='100%'),
            style={'description_width': '0px'}
        )
        
        # Submit button
        self.submit_btn = widgets.Button(
            description='Submit Answer',
            button_style='primary',
            icon='check',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.submit_btn.on_click(self.submit_answer)
        
    def display_question(self):
        """Display current question"""
        with self.question_output:
            clear_output()
            
            if self.current_question >= len(QUESTIONS):
                self.show_results()
                return
            
            q = QUESTIONS[self.current_question]
            
            # Update progress
            self.progress.value = self.current_question
            self.progress.description = f'Question {self.current_question + 1}/{len(QUESTIONS)}'
            
            # Question HTML
            question_html = f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 10px; color: white; margin: 20px 0;'>
                <h2 style='margin: 0 0 10px 0;'>Question {q['id']}</h2>
                <p style='font-size: 18px; line-height: 1.6; margin: 0;'>
                    {q['definition']}
                </p>
                <p style='margin: 15px 0 0 0; font-size: 14px; opacity: 0.9;'>
                    Category: {q['category']}
                </p>
            </div>
            """
            self.question_label.value = question_html
            
            # Update options
            self.radio_options.options = q['options']
            self.radio_options.value = None
            
            # Display components
            display(self.progress)
            display(self.question_label)
            display(self.radio_options)
            display(self.submit_btn)
    
    def submit_answer(self, btn):
        """Handle answer submission"""
        if self.radio_options.value is None:
            with self.output:
                clear_output()
                display(HTML("""
                <div style='background: #fff3cd; padding: 15px; border-radius: 5px; 
                            border-left: 5px solid #ffc107; margin: 10px 0;'>
                    <strong>⚠️ Please select an answer before submitting!</strong>
                </div>
                """))
            return
        
        # Record answer
        q = QUESTIONS[self.current_question]
        user_answer = self.radio_options.value
        is_correct = user_answer == q['answer']
        
        self.user_answers.append({
            'question_id': q['id'],
            'user_answer': user_answer,
            'correct_answer': q['answer'],
            'is_correct': is_correct
        })
        
        if is_correct:
            self.score += 1
        
        # Show feedback
        with self.output:
            clear_output()
            if is_correct:
                display(HTML("""
                <div style='background: #d4edda; padding: 15px; border-radius: 5px; 
                            border-left: 5px solid #28a745; margin: 10px 0;'>
                    <strong style='color: #155724;'>✅ Correct!</strong>
                </div>
                """))
            else:
                display(HTML(f"""
                <div style='background: #f8d7da; padding: 15px; border-radius: 5px; 
                            border-left: 5px solid #dc3545; margin: 10px 0;'>
                    <strong style='color: #721c24;'>❌ Incorrect!</strong><br>
                    <span style='color: #721c24;'>The correct answer is: <strong>{q['answer']}</strong></span>
                </div>
                """))
        
        # Move to next question after short delay
        self.current_question += 1
        
        # Clear output and show next question
        import time
        time.sleep(1.5)
        
        with self.output:
            clear_output()
        
        with self.question_output:
            clear_output()
            self.display_question()
    
    def show_results(self):
        """Display final results"""
        with self.question_output:
            clear_output()
            
            percentage = (self.score / len(QUESTIONS)) * 100
            passed = percentage >= 90  # 90% to pass
            
            # Result summary
            result_html = f"""
            <div style='background: linear-gradient(135deg, {"#4CAF50" if passed else "#f44336"} 0%, {"#81C784" if passed else "#e57373"} 100%); 
                        padding: 40px; border-radius: 15px; color: white; text-align: center; margin: 30px 0;'>
                <h1 style='margin: 0 0 20px 0; font-size: 48px;'>
                    {"🎉 PASSED!" if passed else "❌ FAILED"}
                </h1>
                <h2 style='margin: 0 0 10px 0;'>Your Score: {self.score}/{len(QUESTIONS)}</h2>
                <p style='font-size: 24px; margin: 0;'>Percentage: {percentage:.1f}%</p>
                <p style='margin: 20px 0 0 0; font-size: 16px; opacity: 0.9;'>
                    {"Congratulations! You may proceed to Part 2." if passed else "Please review the COCO categories and try again. You need 80% to pass."}
                </p>
            </div>
            """
            display(HTML(result_html))
            
            # Detailed results
            details_html = """
            <div style='background: white; padding: 25px; border-radius: 10px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0;'>
                <h3 style='margin: 0 0 20px 0; color: #333;'>📊 Detailed Results</h3>
                <table style='width: 100%; border-collapse: collapse;'>
                    <thead>
                        <tr style='background: #f5f5f5;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #ddd;'>Q</th>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #ddd;'>Your Answer</th>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #ddd;'>Correct Answer</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #ddd;'>Result</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for ans in self.user_answers:
                result_icon = "✅" if ans['is_correct'] else "❌"
                row_color = "#f1f8f4" if ans['is_correct'] else "#fef5f5"
                details_html += f"""
                <tr style='background: {row_color};'>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{ans['question_id']}</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{ans['user_answer']}</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{ans['correct_answer']}</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee; text-align: center;'>{result_icon}</td>
                </tr>
                """
            
            details_html += """
                    </tbody>
                </table>
            </div>
            """
            display(HTML(details_html))
    
    def start(self):
        """Start the test"""
        # Welcome screen
        welcome_html = """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 15px; color: white; text-align: center; margin: 30px 0;'>
            <h1 style='margin: 0 0 20px 0; font-size: 42px;'>📝 COCO Label Understanding Test</h1>
            <p style='font-size: 18px; margin: 0 0 10px 0;'>Test your knowledge of COCO dataset categories</p>
            <p style='font-size: 16px; margin: 0; opacity: 0.9;'>30 Questions | 90% Required to Pass</p>
        </div>
        
        <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #1976d2;'>📋 Instructions</h3>
            <ul style='margin: 0; padding-left: 20px; line-height: 1.8;'>
                <li>Read each definition carefully</li>
                <li>Select the correct COCO category from the options</li>
                <li>Click "Submit Answer" to proceed to the next question</li>
                <li>You need to score 90% (27/30) to pass this test</li>
                <li>You will see immediate feedback after each answer</li>
            </ul>
        </div>
        """
        display(HTML(welcome_html))
        
        # Display first question
        with self.question_output:
            self.display_question()
        
        display(self.question_output)
        display(self.output)


# Run the test
if __name__ == "__main__":
    test = COCOQualificationTest()
    test.start()