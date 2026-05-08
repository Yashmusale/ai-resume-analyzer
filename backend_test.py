import requests
import os

BASE_URL = "http://127.0.0.1:5000"

def test_analyze():
    print("Testing /analyze...")
    files = {'resume': open('test_resume.pdf', 'rb')}
    data = {
        'jd': 'Looking for a Senior Backend Developer with expertise in Python, Flask, and SQL. Experience with AWS and Docker is a plus.',
        'role': 'Backend Developer'
    }
    response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Match Score: {result.get('match_score')}")
    print(f"ATS Score: {result.get('ats_score')}")
    print(f"AI Feedback Preview: {result.get('ai_feedback')[:100]}...")
    return result

def test_rewrite(resume_text, jd):
    print("\nTesting /rewrite...")
    data = {
        'resume_text': resume_text,
        'jd_text': jd
    }
    response = requests.post(f"{BASE_URL}/rewrite", json=data)
    print(f"Status: {response.status_code}")
    print(f"Rewritten Content Preview: {response.json().get('rewritten_content')[:100]}...")

def test_questions(resume_text, jd):
    print("\nTesting /generate_questions...")
    data = {
        'resume_text': resume_text,
        'jd_text': jd
    }
    response = requests.post(f"{BASE_URL}/generate_questions", json=data)
    print(f"Status: {response.status_code}")
    print(f"Questions Preview: {response.json().get('questions')[:100]}...")

if __name__ == "__main__":
    try:
        jd = 'Looking for a Senior Backend Developer with expertise in Python, Flask, and SQL. Experience with AWS and Docker is a plus.'
        analysis = test_analyze()
        resume_text = analysis.get('resume_text')
        test_rewrite(resume_text, jd)
        test_questions(resume_text, jd)
    except Exception as e:
        print(f"Error: {e}")
