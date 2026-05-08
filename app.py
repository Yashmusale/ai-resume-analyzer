import os
import pdfplumber
import re
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from fpdf import FPDF
import io
from flask import send_file

load_dotenv(override=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Model
class JobApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(50), default='Applied')
    date = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# Find .env file absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path, override=True)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        with open(dotenv_path, 'r') as f:
            for line in f:
                if 'GEMINI_API_KEY' in line:
                    GEMINI_API_KEY = line.split('=')[1].strip()
    except Exception:
        pass

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"--- Gemini API Configured Successfully [Key: {GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-4:]}] ---")
else:
    print("--- CRITICAL ERROR: GEMINI_API_KEY NOT FOUND ---")

# Role-specific priority skills
ROLE_SKILLS = {
    'Data Analyst': ['SQL', 'Excel', 'Tableau', 'Power BI', 'Pandas', 'NumPy', 'Statistics', 'Python'],
    'AI Engineer': ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Python', 'Scikit-learn', 'NLP'],
    'Backend Developer': ['Node.js', 'Flask', 'Django', 'SQL', 'NoSQL', 'PostgreSQL', 'Docker', 'Kubernetes', 'REST API']
}

# List of common technical skills (can be expanded)
SKILLS_DB = [
    'Python', 'Java', 'C++', 'JavaScript', 'React', 'Angular', 'Vue', 'Node.js', 
    'Express', 'Flask', 'Django', 'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL',
    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD', 'Git', 'GitHub',
    'Machine Learning', 'Deep Learning', 'Data Science', 'Pandas', 'NumPy', 
    'Scikit-learn', 'TensorFlow', 'PyTorch', 'Tableau', 'Power BI', 'Excel',
    'HTML', 'CSS', 'Tailwind', 'Bootstrap', 'REST API', 'GraphQL', 'Microservices',
    'Agile', 'Scrum', 'Project Management', 'Communication', 'Teamwork'
]

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

def extract_skills(text):
    skills = []
    for skill in SKILLS_DB:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            skills.append(skill)
    return skills

def calculate_similarity(resume_text, jd_text):
    try:
        if not resume_text.strip() or not jd_text.strip():
            return 0.0
        documents = [resume_text, jd_text]
        count_vectorizer = TfidfVectorizer(stop_words='english')
        sparse_matrix = count_vectorizer.fit_transform(documents)
        similarity_matrix = cosine_similarity(sparse_matrix)
        return float(similarity_matrix[0][1] * 100)
    except Exception as e:
        print(f"Similarity Calculation Error: {e}")
        return 0.0

def calculate_ats_score(resume_text, jd_skills, resume_skills, role=None):
    if not jd_skills:
        skill_score = 0
    else:
        matched_skills = set(resume_skills).intersection(set(jd_skills))
        skill_score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
    
    # Role-based boost
    role_boost = 0
    if role and role in ROLE_SKILLS:
        target_skills = set(ROLE_SKILLS[role])
        found_target = set(resume_skills).intersection(target_skills)
        role_boost = (len(found_target) / len(target_skills)) * 20 # Up to 20% boost for role-specific skills
    
    word_count = len(resume_text.split())
    length_score = 100 if 400 <= word_count <= 1000 else 70
    
    final_score = (skill_score * 0.6) + (length_score * 0.2) + (role_boost)
    return min(float(final_score), 100.0)

def generate_ai_feedback(resume_text, jd_text, role=None):
    if not GEMINI_API_KEY:
        return "AI Feedback unavailable: API key not configured."
    
    try:
        resume_truncated = resume_text[:4000]
        jd_truncated = jd_text[:4000]
        role_context = f"\nTarget Role: {role}" if role else ""
        
        # Try multiple model names for compatibility
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
        last_error = ""
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
                As an expert HR Recruiter and ATS Specialist, analyze the following resume against the job description for a {role if role else 'specific'} position.{role_context}
                
                Resume Content:
                {resume_truncated}
                
                Job Description:
                {jd_truncated}
                
                Please provide:
                1. **Strengths**: What matches well for this {role if role else 'role'}?
                2. **Weaknesses**: What specific {role if role else 'role'}-based skills are missing?
                3. **Improvements**: Specific actionable advice to increase the ATS score for this career path.
                
                Format the response in clean Markdown.
                """
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = str(e)
                continue
        
        return f"Error generating AI feedback: {last_error}"
    except Exception as e:
        return f"Error during feedback generation: {str(e)}"

def rewrite_resume_content(resume_text, jd_text=None):
    if not GEMINI_API_KEY:
        return "AI Rewriting unavailable: API key not configured."
    
    try:
        resume_truncated = resume_text[:5000]
        jd_context = f"\nTarget Job Description:\n{jd_text[:2000]}" if jd_text else ""
        
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
        last_error = ""
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
                As a professional Resume Writer, rewrite the following resume content to make it more impactful and ATS-friendly.
                
                Original Resume:
                {resume_truncated}
                {jd_context}
                
                Please provide:
                1. **Professional Summary**: A compelling summary that highlights key strengths.
                2. **Enhanced Experience**: Rewrite bullet points using the Action-Verb + Task + Result (STAR) method. 
                3. **Refined Tone**: Use industry-standard professional wording.
                
                Format the output in clean Markdown.
                """
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = str(e)
                continue
        return f"Error during rewriting: {last_error}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_interview_questions(resume_text, jd_text):
    if not GEMINI_API_KEY:
        return "Interview Questions unavailable: API key not configured."
    
    try:
        resume_truncated = resume_text[:4000]
        jd_truncated = jd_text[:4000]
        
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
        last_error = ""
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
                Based on the candidate's resume and the job description provided, generate a list of targeted interview questions.
                
                Resume:
                {resume_truncated}
                
                Job Description:
                {jd_truncated}
                
                Please provide exactly:
                1. **5 Technical Questions**: Focused on the skills required for the role and mentioned in the resume.
                2. **3 HR/Behavioral Questions**: To assess cultural fit and soft skills.
                3. **2 Project-Based Questions**: Deep diving into projects mentioned in the resume.
                
                Format the output in clean Markdown with clear headings.
                """
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = str(e)
                continue
        return f"Error generating questions: {last_error}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400
    
    resume_file = request.files['resume']
    jd_text = request.form.get('jd', '')
    role = request.form.get('role', 'Default')
    
    if resume_file.filename == '':
        return jsonify({"error": "Empty file name"}), 400
    
    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)
    
    try:
        print(f"--- Analysis Started for {resume_file.filename} ---")
        resume_text = extract_text_from_pdf(resume_path)
        print(f"Text Extracted ({len(resume_text)} chars)")
        
        if not resume_text.strip():
            print("Error: No text found in PDF")
            return jsonify({"error": "Could not extract text from PDF. It might be scanned or empty."}), 400

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        print(f"Skills Extracted: {len(resume_skills)} from Resume, {len(jd_skills)} from JD")
        
        match_score = calculate_similarity(resume_text, jd_text)
        ats_score = calculate_ats_score(resume_text, jd_skills, resume_skills, role)
        print(f"Scores Calculated: Match={match_score}, ATS={ats_score}")
        
        missing_skills = list(set(jd_skills) - set(resume_skills))
        
        # Get AI Feedback
        print("Requesting AI Feedback...")
        ai_feedback = generate_ai_feedback(resume_text, jd_text, role)
        print("AI Feedback Received")
        
        print("--- Analysis Completed Successfully ---")
        return jsonify({
            "match_score": round(match_score, 2),
            "ats_score": round(ats_score, 2),
            "resume_skills": resume_skills,
            "missing_skills": missing_skills,
            "ai_feedback": ai_feedback,
            "resume_text": resume_text
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Analysis Error: {str(e)}"}), 500
    finally:
        if os.path.exists(resume_path):
            os.remove(resume_path)

@app.route('/rewrite', methods=['POST'])
def rewrite():
    data = request.json
    resume_text = data.get('resume_text', '')
    jd_text = data.get('jd_text', '')
    
    if not resume_text:
        return jsonify({"error": "Resume text is missing"}), 400
        
    rewritten_content = rewrite_resume_content(resume_text, jd_text)
    return jsonify({"rewritten_content": rewritten_content})

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.json
    resume_text = data.get('resume_text', '')
    jd_text = data.get('jd_text', '')
    
    if not resume_text:
        return jsonify({"error": "Resume text is missing"}), 400
        
    questions = generate_interview_questions(resume_text, jd_text)
    return jsonify({"questions": questions})

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.json
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "Resume Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    # Scores
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(99, 102, 241) # Primary color
    pdf.cell(0, 10, f"Match Score: {data['match_score']}%", ln=True)
    pdf.cell(0, 10, f"ATS Score: {data['ats_score']}%", ln=True)
    pdf.ln(5)
    
    # Skills
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Extracted Skills:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 8, ", ".join(data['resume_skills']))
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Missing Skills:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(239, 68, 68) # Red
    pdf.multi_cell(0, 8, ", ".join(data['missing_skills']) if data['missing_skills'] else "None")
    pdf.ln(5)
    
    # AI Feedback
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "AI Feedback & Suggestions:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    # Basic cleaning of markdown symbols and encoding safety for PDF
    feedback = data['ai_feedback'].replace('**', '').replace('###', '').replace('*', '-').replace('`', "'")
    # FPDF1/2 Helvetica only supports Latin-1. Strip other characters to prevent crash.
    feedback_safe = feedback.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 8, feedback_safe)
    
    output = io.BytesIO()
    # fpdf2 output(dest='S') returns bytes
    pdf_bytes = pdf.output()
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')
    output.write(pdf_bytes)
    output.seek(0)
    
    return send_file(
        output,
        as_attachment=True,
        download_name="Resume_Analysis_Report.pdf",
        mimetype="application/pdf"
    )

@app.route('/jobs', methods=['GET'])
def get_jobs():
    jobs = JobApplication.query.order_by(JobApplication.date.desc()).all()
    return jsonify([{
        "id": job.id,
        "company": job.company,
        "role": job.role,
        "status": job.status,
        "date": job.date.strftime('%Y-%m-%d')
    } for job in jobs])

@app.route('/jobs', methods=['POST'])
def add_job():
    data = request.json
    new_job = JobApplication(
        company=data['company'],
        role=data['role'],
        status=data.get('status', 'Applied')
    )
    db.session.add(new_job)
    db.session.commit()
    return jsonify({"message": "Job added successfully"})

@app.route('/jobs/<int:id>', methods=['PUT'])
def update_job(id):
    job = JobApplication.query.get_or_404(id)
    data = request.json
    if 'status' in data:
        job.status = data['status']
    db.session.commit()
    return jsonify({"message": "Job updated successfully"})

@app.route('/jobs/<int:id>', methods=['DELETE'])
def delete_job(id):
    job = JobApplication.query.get_or_404(id)
    db.session.delete(job)
    db.session.commit()
    return jsonify({"message": "Job deleted successfully"})

if __name__ == '__main__':
    app.run(debug=True)
