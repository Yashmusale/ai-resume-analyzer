# AI Resume Analyzer

A professional full-stack web application built with Python Flask to analyze resumes against job descriptions using NLP and machine learning.

## Features
- **PDF Extraction**: Uses `pdfplumber` for high-quality text extraction.
- **Skill Matching**: Automatically identifies technical skills in both the resume and JD.
- **Match Score**: Calculates semantic similarity using TF-IDF and Cosine Similarity.
- **ATS Score**: Heuristic-based score evaluating keyword density and document formatting.
- **Interactive UI**: Modern dashboard with animated progress charts and glassmorphism.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in your browser.

## Technologies Used
- **Backend**: Flask, Scikit-learn, pdfplumber
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript
- **NLP**: TF-IDF Vectorization, Regex-based skill extraction


## Screenshots

![Screenshot](screenshot/1.png)

![Screenshot](screenshot/2.png)

![Screenshot](screenshot/3.png)

![Screenshot](screenshot/4.png)

![Screenshot](screenshot/5.png)

![Screenshot](screenshot/6.png)

![Screenshot](screenshot/7.png)

![Screenshot](screenshot/8.png)
