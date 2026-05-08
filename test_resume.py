from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(40, 10, "John Doe - Software Engineer")
pdf.ln(10)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, "Summary: Experienced developer with 5 years in Python and Backend development. Expert in Flask, SQL, and AWS.\n\nSkills: Python, Flask, SQL, AWS, Docker, Git, JavaScript.\n\nExperience: Senior Developer at TechCorp. Built scalable microservices using Python.")
pdf.output("test_resume.pdf")
