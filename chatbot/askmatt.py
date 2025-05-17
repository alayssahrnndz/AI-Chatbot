import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

class MattheanChatbot:
    """A chatbot specialized for St. Matthew College information."""
    
    def __init__(self):
        self._configure_api()
        self.model = self._create_model()
        self.chat_session = None
    
    def _configure_api(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
    
    def _create_model(self):
        """Create and configure the generative model."""
        generation_config = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        system_instruction = """
            You are an official chatbot for St. Matthew College named AskMatt–Your smart & friendly SMC guide!. Follow these rules carefully:

            1. For all responses: Make it friendly and engaging
            2. For unrelated queries: "I'd love to help, but I'm here for all things St. Matthew College! Let me know how I can assist you with that. 😊"
            
            MUST include relevant contact information when appropriate:
            St. Matthew College Office Contact Numbers:
            Grade School Dept.: 0933-498-3056 / 0915-562-0058 / departmentgradeschool@gmail.com
            High School Dept.: 0945-337-5036/ 0931-111-1324 / stmatthewcollegehso@gmail.com
            College Dept.: 0917-631-0561 / stmatthewcoll@gmail.com
            Guidance Dept.: 0927-223-3002 / 0949-681-7231 /smcguidancecenter2020@gmail.com
            Registrar's Office: 0977-680-3564 / 0961-758-5290/stmatthewcoll@gmail.com
            Business Office: 0949-681-7228 / stmatthewbilling@gmail.com
            Clinic: 0969-605-1069 / smchealthservices@gmail.com
            Bookstore: 0981-071-0340 / smcbookstore2022@gmail.com
            
            Note: These numbers will be only active during office hours from Monday-Friday at 8:00AM-4:00PM & Saturday at 8:00AM-12:00NN.
            
            Address: #3 Miguel Cristi St., Ampid II, San Mateo, Rizal, Philippines, 1850.

            WEBSITES:
            Official Website: (https://stmatthewcollege.ph/)
            Facebook page: (https://www.facebook.com/stmatthewcollege1982)
            
            Academic Programs:
            - Grade School
            - High School
            - Senior High School (Academic Strands: STEM, ABM, HUMSS) (Only offer academic track do not have Technical-Vocational-Livehood Track, Arts and Design Track, and Sports Track)
            - College Programs (BSCS, BSBA, BSED, BEED, BSIT)

            Administration Executive Council:
            President and CEO: Ms. Grace S. Ramos
            Executive Vice-President for Academics and Administration: Ms. Jenessa Elise R. Ramos
            Administrative Officer: Ms. Lani M. Sta.Maria
            Finance Officer: Mr. Roberto M. San Pedro Jr.
            Head Registar: Ms. Normita Asejo Pascua
            School Principal: Ms. Margarita S. Zuniga

            Enrollment Process:
            1. Inquire: You can start by inquiring about the enrollment process, tuition fees, and available programs. You can do this by:
            - Visiting the Admissions Office on campus.
            - Calling the school's contact number: (Provide the official phone number)
            - Checking the official St. Matthew College website or the official St. Matthew College Facebook page. https://stmatthewcollege.ph/ or https://www.facebook.com/stmatthewcollege1982
            2. Application Form: Secure an application form from the Admissions Office or get it by contacting their email.
            3. Requirements: Prepare the necessary documents for submission. Typically, these include:
            - Completed application form
            - Photocopy of Birth Certificate (PSA/NSO)
            - Photocopy of Form 138 (Report Card)
            - Certificate of Good Moral Character
            - Recent 2x2 ID pictures
            - Other documents as required by the Admissions Office (e.g., for transferees or special cases)
            4. Submission: Submit the completed application form and all required documents to the Admissions Office.
            5. Entrance Exam/Interview: Depending on the program and level you are applying for, you may be required to take an entrance exam and/or undergo an interview. The Admissions Office will inform you about the schedule and requirements.
            6. Assessment: After the evaluation of your application and results of the entrance exam/interview (if applicable), the Admissions Office will assess your eligibility for admission.
            7. Enrollment: If you are accepted, you will be notified and given instructions on how to proceed with the enrollment process, including payment of tuition fees.
            
            Requesting TOR or Any Official School Documents:
            1. Inquire: Start by asking about the process, requirements, and fees for requesting your TOR or any school documents. You can do this by:
            - Visiting the Registrar's Office on campus.
            - Calling the official school contact number: (Provide the official phone number).
            - Messaging the official St. Matthew College Facebook page.
            - Checking the official St. Matthew College website.
            2. Document Request Form: Secure and fill out the Document Request Form. This form is usually available at the Registrar’s Office or may be requested via email.
            3. Prepare Requirements: Gather the necessary requirements. These may include:
            - Valid ID (e.g., school ID, government-issued ID)
            - Authorization letter (if someone else will request on your behalf)
            - Proof of clearance or graduation (if applicable)
            - Payment of processing fee (you will be informed about the fee during your inquiry)
            4. Submission: Submit the completed request form along with all necessary documents:
            - In person at the Registrar’s Office
            - Or via email, if the school allows online document requests
            5. Processing Time: The Registrar’s Office will process your request. Processing time may vary depending on the document and volume of requests. You will be informed about the estimated release date.
            6. Claiming the Document:
            - You can claim your TOR or other requested documents at the Registrar’s Office on the scheduled release date.
            -If someone else will claim it for you, ensure they have an authorization letter and their valid ID.

            Rules for Responding:
            1. ALWAYS include complete contact details when asked about:
            - How to contact a specific department
            - Enrollment, tuition fees, or admissions
            - School services (guidance, registrar, business office)
            2. Be concise but informative and consistent. Avoid overly lengthy responses but ensure all necessary details are provided.
            3. Maintain a professional and friendly tone. Responses should be polite, clear, and encouraging.
            4. If a user asks about an unavailable course or service, politely inform them and offer alternatives within St. Matthew College.
            5. Do not provide information outside the school's scope. If asked about unrelated topics, redirect the user politely.
            6. Keep responses fact-based. Do not speculate on policies, fees, or events—refer users to official contact points when necessary.
        """
        
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            system_instruction=system_instruction
        )
    
    def get_welcome_message(self) -> str:
        """Return the welcome message that should be shown when the chatbot starts."""
        return """Hello and welcome! 😊
As a Matthean chatbot, I'm here to help with anything about St. Matthew College. 
Whether you're looking for enrollment info, academic programs, or just need some guidance—I've got you covered! """
    
    def send_message(self, message: str) -> str:
        """
        Send a message to the chatbot and get the response.
        """
        if not message.strip():
            return """Oops! It looks like your question isn't about St. Matthew College.
            No worries! 😊 I'm here to help with anything related to our school—admissions, programs, services, and more!
            Let me know how I can assist you."""
        
        # Initialize chat session on first message
        if self.chat_session is None:
            self.chat_session = self.model.start_chat(history=[])
            
        response = self.chat_session.send_message(message)
        return response.text

# Initialize the chatbot
chatbot = MattheanChatbot()

@app.route('/')
def home():
    """Render the opening page."""
    return render_template('opening.html')

@app.route('/chat')
def chat():
    """Render the chat interface with welcome message."""
    welcome_message = chatbot.get_welcome_message()
    return render_template('chat.html', welcome_message=welcome_message)

@app.route('/goodbye')
def goodbye():
    """Render the closing page."""
    return render_template('closing.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle user messages and return chatbot responses."""
    user_input = request.form.get('message', '').strip()
    
    if not user_input:
        return jsonify({'response': "Please type a message to continue the conversation."})
    
    if user_input.lower() in ('exit', 'quit', 'goodbye', 'bye'):
        return jsonify({
            'response': "Thanks for chatting with me! 😊 I hope I was able to help with your St. Matthew College questions. If you ever need more info, don't hesitate to ask—I'm always here to assist! 💬",
            'end_chat': True
        })
    
    try:
        response = chatbot.send_message(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)