# Data Dictionary — Student Performance Prediction Platform

**Platform**: Student Performance Prediction Platform  
**Domain**: Educational Data Mining (EDM) & Continuous Academic Evaluation  
**Sustainable Development Goal**: SDG 4 — Quality Education  

---

## Overview

The dataset contains **500 anonymized student academic and behavioral records** reflecting internal continuous evaluations, attendance, study habits, and historical academic background. These features are used to classify a student's risk level (**High Risk**, **Medium Risk**, **Low Risk**) and predict academic performance for proactive mentor intervention.

---

## Feature Descriptions

| Feature Name | Data Type | Range / Values | Description | Role / Significance |
|:---|:---:|:---:|:---|:---|
| `student_id` | String | `STD001` to `STD500` | Unique anonymized identifier for each student | Metadata / Primary Key |
| `name` | String | Anonymized names | Student full name for display in teacher directory | Metadata |
| `attendance` | Float | `40.0%` – `100.0%` | Overall lecture & lab attendance percentage | **Key Predictor**: Attendance < 75% triggers regulatory & academic alerts |
| `internal_marks` | Float | `15.0` – `50.0` | Sessional / internal test marks (scaled out of 50) | Direct metric of subject comprehension and class engagement |
| `assignment_rate` | Float | `30.0%` – `100.0%` | Percentage of assignments submitted on time | Measure of continuous effort and self-discipline |
| `study_hours` | Float | `2.0` – `28.0` | Weekly self-study hours outside class schedule | Work ethic indicator |
| `class_test_marks` | Float | `8.0` – `30.0` | Periodic unit tests and quiz average (out of 30) | Short-term learning retention |
| `past_gpa` | Float | `4.2` – `9.8` | Cumulative Grade Point Average of prior semesters (out of 10.0) | Historical academic baseline |
| `parental_education` | Categorical | `High School`, `Bachelor`, `Master` | Highest parental educational qualification | Socio-demographic context |
| `extracurricular` | Categorical / Binary | `0` (No), `1` (Yes) | Active participation in college clubs/sports | Holistic engagement |
| `internet_access` | Categorical / Binary | `0` (No), `1` (Yes) | Reliable high-speed internet availability at home/hostel | Resource accessibility |
| `risk_level` | Categorical (Target) | `Low Risk`, `Medium Risk`, `High Risk` | Multi-class risk categorization for targeted interventions | Primary prediction target |
| `final_grade_category` | Categorical | `Excellent`, `Good`, `Pass`, `At-Risk` | Qualitative final projected academic tier | Secondary classification / reporting target |
| `predicted_cluster` | Categorical | `Cluster 0 (High)`, `Cluster 1 (Medium)`, `Cluster 2 (Low)` | Unsupervised K-Means clustering assignment (Module 3) | Student grouping for mentor allocation |

---

## Risk Level Definitions

1. **High Risk (At-Risk)**:
   - High likelihood of failing or getting semester back / detention.
   - Typically characterized by `attendance < 72%`, `internal_marks < 25/50`, and `assignment_rate < 60%`.
   - **Recommended Action**: Immediate mentor counseling session, mandatory remedial classes, parental notification.

2. **Medium Risk (Borderline)**:
   - Capable of passing but showing vulnerability in select areas (e.g. erratic attendance or weak class tests).
   - **Recommended Action**: Peer-assisted study groups, weekly assignment check-in, doubt clearing workshops.

3. **Low Risk (Safe / High Performer)**:
   - High attendance (>80%), solid internal marks, consistent assignment submissions, and high past GPA.
   - **Recommended Action**: Advanced seminar opportunities, hackathons, competitive exam / research guidance.

---

## Preprocessing & Feature Engineering Applied

- **Imputation**: No missing entries in final dataset (standardized pre-check).
- **Categorical Encoding**: One-Hot / Label encoding for categorical variables (`parental_education`, binary flags).
- **Feature Scaling**: `StandardScaler` applied to continuous variables for distance-based models (MLP, Logistic Regression, K-Means).
- **Class Imbalance Handling**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** with stratified k-fold split to elevate minority `High Risk` recall from ~55% to >80%.
