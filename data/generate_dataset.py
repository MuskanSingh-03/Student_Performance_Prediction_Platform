"""
Dataset Generator for Student Performance Prediction Platform
Generates 500 realistic, anonymized student records based on continuous evaluation metrics.
"""

import numpy as np
import pandas as pd
import random
import os

np.random.seed(42)
random.seed(42)

FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Aniket", "Anjali", "Arjun", "Ayush",
    "Bhavna", "Chirag", "Deepak", "Divya", "Gaurav", "Harsh", "Isha", "Ishaan",
    "Jatin", "Kavya", "Khushi", "Kunal", "Manish", "Meera", "Mohit", "Muskan",
    "Neha", "Nicky", "Nidhi", "Nikhil", "Nivedita", "Pooja", "Pragati", "Pranay",
    "Priya", "Rahul", "Rhea", "Rishabh", "Riya", "Rohan", "Sakshi", "Sameer",
    "Sanya", "Shashank", "Shivam", "Shruti", "Siddharth", "Simran", "Sneha",
    "Tanvi", "Utkarsh", "Varun", "Vikas", "Vishal", "Yash", "Zoya"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Yadav", "Tomar", "Gola", "Mishra",
    "Kumar", "Pandey", "Chauhan", "Saxena", "Agarwal", "Tiwari", "Bhatnagar",
    "Dubey", "Shukla", "Joshi", "Choudhary", "Tripathi", "Rawat", "Malhotra"
]

def generate_students(n=500):
    records = []
    
    for i in range(1, n + 1):
        std_id = f"STD{i:03d}"
        full_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        
        # Socio-demographic & contextual features
        parental_edu = np.random.choice(["High School", "Bachelor", "Master"], p=[0.30, 0.52, 0.18])
        internet_access = np.random.choice([1, 0], p=[0.92, 0.08])
        extracurricular = np.random.choice([1, 0], p=[0.48, 0.52])
        
        # Base academic capability latent variable (0 to 1)
        latent_ability = np.clip(np.random.normal(0.62, 0.18), 0.10, 0.95)
        
        # Correlated academic features
        attendance = np.clip(latent_ability * 85 + np.random.normal(15, 8), 42.0, 99.5)
        attendance = round(float(attendance), 1)
        
        internal_marks = np.clip(latent_ability * 42 + np.random.normal(5, 4), 14.0, 49.5)
        internal_marks = round(float(internal_marks), 1)
        
        class_test_marks = np.clip(latent_ability * 25 + np.random.normal(3.5, 2.5), 8.0, 29.8)
        class_test_marks = round(float(class_test_marks), 1)
        
        assignment_rate = np.clip(latent_ability * 80 + np.random.normal(18, 9), 32.0, 100.0)
        assignment_rate = round(float(assignment_rate), 1)
        
        study_hours = np.clip(latent_ability * 18 + np.random.normal(4, 3.5), 2.0, 26.0)
        study_hours = round(float(study_hours), 1)
        
        past_gpa = np.clip(latent_ability * 5.2 + np.random.normal(4.2, 0.6), 4.2, 9.85)
        past_gpa = round(float(past_gpa), 2)
        
        # Composite score determining academic health & risk
        # Attendance (30%), Internal marks (30%), Past GPA (20%), Assignment rate (10%), Study hours (10%)
        composite_score = (
            0.30 * (attendance / 100.0) +
            0.30 * (internal_marks / 50.0) +
            0.20 * (past_gpa / 10.0) +
            0.10 * (assignment_rate / 100.0) +
            0.10 * (study_hours / 25.0)
        )
        
        # Real-world class distribution: High Risk is minority (~18-20%), Medium (~30-35%), Low Risk (~48-50%)
        # Composite score: higher means better performance
        if composite_score < 0.44 or (attendance < 60.0 and internal_marks < 24.0):
            risk_level = "High Risk"
            grade = "At-Risk"
        elif composite_score < 0.65:
            risk_level = "Medium Risk"
            grade = "Pass" if composite_score < 0.55 else "Good"
        else:
            risk_level = "Low Risk"
            grade = "Excellent"
            
        records.append({
            "student_id": std_id,
            "name": full_name,
            "attendance": attendance,
            "internal_marks": internal_marks,
            "assignment_rate": assignment_rate,
            "study_hours": study_hours,
            "class_test_marks": class_test_marks,
            "past_gpa": past_gpa,
            "parental_education": parental_edu,
            "extracurricular": int(extracurricular),
            "internet_access": int(internet_access),
            "risk_level": risk_level,
            "final_grade": grade
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_students(500)
    csv_path = os.path.join(out_dir, "students_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records to {csv_path}")
    print("Risk distribution:")
    print(df["risk_level"].value_counts(normalize=True))
