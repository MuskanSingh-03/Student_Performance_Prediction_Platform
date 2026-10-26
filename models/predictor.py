"""
Inference & Early Warning Diagnostic Engine
Integrates trained Random Forest (primary), MLP, Decision Tree, Logistic Regression models,
and K-Means clustering with risk reasoning and actionable intervention protocols.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
DATA_DIR = os.path.join(BASE_DIR, "data")

def ensure_artifacts_exist():
    """Ensures dataset and trained models exist; generates/trains if missing on deployment."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    csv_path = os.path.join(DATA_DIR, "students_dataset.csv")
    rf_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    
    if not os.path.exists(csv_path):
        print("[Deployment] Dataset missing. Generating 500 student records...")
        from data.generate_dataset import generate_students
        df = generate_students(500)
        df.to_csv(csv_path, index=False)
        
    if not os.path.exists(rf_path) or not os.path.exists(metrics_path):
        print("[Deployment] Trained models missing. Auto-running training pipeline...")
        from models.train_models import train_and_evaluate
        train_and_evaluate()

ensure_artifacts_exist()

# Load models and preprocessing artifacts
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
mlp_model = joblib.load(os.path.join(MODELS_DIR, "mlp_model.joblib"))
dt_model = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.joblib"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.joblib"))
kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

with open(os.path.join(MODELS_DIR, "model_metrics.json"), "r") as f:
    MODEL_METRICS = json.load(f)

FEATURE_COLS = [
    "attendance", "internal_marks", "assignment_rate", "study_hours",
    "class_test_marks", "past_gpa", "parental_education",
    "extracurricular", "internet_access"
]

EDU_MAP = {"High School": 0, "Bachelor": 1, "Master": 2}

def preprocess_single_input(data_dict):
    """Encodes and scales a single student feature dictionary into model input array."""
    raw_edu = data_dict.get("parental_education", "Bachelor")
    edu_val = EDU_MAP.get(raw_edu, 1)
    
    features = [
        float(data_dict.get("attendance", 75.0)),
        float(data_dict.get("internal_marks", 35.0)),
        float(data_dict.get("assignment_rate", 80.0)),
        float(data_dict.get("study_hours", 12.0)),
        float(data_dict.get("class_test_marks", 20.0)),
        float(data_dict.get("past_gpa", 7.0)),
        edu_val,
        int(data_dict.get("extracurricular", 0)),
        int(data_dict.get("internet_access", 1))
    ]
    
    features_df = pd.DataFrame([features], columns=FEATURE_COLS)
    scaled = scaler.transform(features_df)
    return scaled, features_df

def compute_early_warnings(student_data):
    """
    Evaluates key academic indicators against university thresholds
    and returns human-readable risk triggers.
    """
    warnings = []
    attendance = float(student_data.get("attendance", 100))
    internal = float(student_data.get("internal_marks", 50))
    assignment = float(student_data.get("assignment_rate", 100))
    study_hours = float(student_data.get("study_hours", 15))
    past_gpa = float(student_data.get("past_gpa", 8.0))
    class_test = float(student_data.get("class_test_marks", 25))
    
    if attendance < 65.0:
        warnings.append({
            "type": "critical",
            "title": "Severe Attendance Shortage",
            "detail": f"Attendance is {attendance:.1f}% (< 65%). High probability of semester exam detention under college guidelines."
        })
    elif attendance < 75.0:
        warnings.append({
            "type": "warning",
            "title": "Attendance Alert",
            "detail": f"Attendance is {attendance:.1f}% (< 75%). Below mandatory minimum threshold."
        })
        
    if internal < 25.0:
        warnings.append({
            "type": "critical",
            "title": "Sub-par Internal Assessment",
            "detail": f"Internal mark is {internal:.1f}/50 (< 50% passing mark). Critical subject comprehension gap."
        })
    elif internal < 32.0:
        warnings.append({
            "type": "warning",
            "title": "Average Internal Marks",
            "detail": f"Internal score {internal:.1f}/50 needs improvement to secure safe SGPA."
        })
        
    if assignment < 60.0:
        warnings.append({
            "type": "warning",
            "title": "Low Assignment Submission",
            "detail": f"Assignment submission rate is {assignment:.1f}%. Indicates inconsistent coursework submission."
        })
        
    if study_hours < 7.0:
        warnings.append({
            "type": "info",
            "title": "Low Self-Study Allocation",
            "detail": f"Weekly self-study is only {study_hours:.1f} hrs/week (< 1 hr/day). Additional study time recommended."
        })
        
    if past_gpa < 6.0:
        warnings.append({
            "type": "critical",
            "title": "Historical Academic Vulnerability",
            "detail": f"Cumulative GPA is {past_gpa:.2f}/10.0, indicating persistent academic struggle across prior terms."
        })
        
    return warnings

def generate_intervention_strategy(risk_level, warnings):
    """Generates tailored mentor action items based on predicted risk level and specific failure modes."""
    if risk_level == "High Risk":
        return {
            "tier": "Tier 1: Immediate Critical Intervention",
            "badge_color": "danger",
            "actions": [
                "Schedule one-on-one Faculty Mentor counseling session within 48 hours.",
                "Enroll in mandatory departmental remedial classes and peer buddy study group.",
                "Send formal attendance and continuous assessment notice to student and guardian.",
                "Assign a bi-weekly academic progress monitor to track assignment catch-up."
            ]
        }
    elif risk_level == "Medium Risk":
        return {
            "tier": "Tier 2: Guided Support & Regular Monitoring",
            "badge_color": "warning",
            "actions": [
                "Conduct a 15-minute mid-term review with subject course coordinator.",
                "Recommend guided problem-solving tutorial sheets for weak topics.",
                "Set weekly assignment completion milestones with teaching assistants.",
                "Encourage participation in technical study circles and doubt-clearing sessions."
            ]
        }
    else:
        return {
            "tier": "Tier 3: Enrichment & Advanced Opportunities",
            "badge_color": "success",
            "actions": [
                "Nominate for student peer tutor or departmental technical lead roles.",
                "Encourage participation in hackathons, research publications, or GATE/GRE prep.",
                "Offer advanced elective seminars or industry certification pathways.",
                "Maintain current study routine and exam readiness habits."
            ]
        }

def predict_single(data_dict, model_choice="Random Forest"):
    """
    Executes prediction for a single student profile.
    Returns predicted risk level, probability distribution, all-model comparison,
    cluster segment, early warnings, and recommended mentor interventions.
    """
    scaled_features, raw_df = preprocess_single_input(data_dict)
    
    # Selected model prediction
    model_map = {
        "Random Forest": rf_model,
        "MLP Neural Network": mlp_model,
        "Decision Tree": dt_model,
        "Logistic Regression": lr_model
    }
    
    chosen_model = model_map.get(model_choice, rf_model)
    pred_idx = chosen_model.predict(scaled_features)[0]
    pred_label = label_encoder.classes_[pred_idx]
    
    # Prediction probabilities
    if hasattr(chosen_model, "predict_proba"):
        probabilities = chosen_model.predict_proba(scaled_features)[0]
        prob_dict = {
            label_encoder.classes_[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(label_encoder.classes_))
        }
        confidence = float(max(probabilities)) * 100
    else:
        prob_dict = {pred_label: 100.0}
        confidence = 100.0
        
    # Predictions across all 4 models for comparison
    all_predictions = {}
    for name, mdl in model_map.items():
        idx = mdl.predict(scaled_features)[0]
        all_predictions[name] = label_encoder.classes_[idx]
        
    # K-Means Cluster assignment
    cluster_idx = kmeans_model.predict(scaled_features)[0]
    cluster_label_map = {
        0: "High Performers (Consistent)",
        1: "Moderate Performers (Developing)",
        2: "At-Risk / Low Performers"
    }
    cluster_name = cluster_label_map.get(cluster_idx, f"Cluster {cluster_idx}")
    
    # Calculate projected GPA estimate (heuristic regression projection)
    att = float(data_dict.get("attendance", 75))
    int_m = float(data_dict.get("internal_marks", 35))
    past_gpa = float(data_dict.get("past_gpa", 7.0))
    proj_gpa = round(0.4 * past_gpa + 0.35 * (int_m / 5.0) + 0.25 * (att / 10.0), 2)
    proj_gpa = min(10.0, max(3.5, proj_gpa))
    
    # Early Warnings & Intervention Strategy
    early_warnings = compute_early_warnings(data_dict)
    intervention = generate_intervention_strategy(pred_label, early_warnings)
    
    return {
        "predicted_risk": pred_label,
        "model_used": model_choice,
        "confidence": round(confidence, 1),
        "probabilities": prob_dict,
        "all_model_predictions": all_predictions,
        "cluster": cluster_name,
        "projected_gpa": proj_gpa,
        "early_warnings": early_warnings,
        "intervention": intervention,
        "input_features": data_dict
    }

def predict_batch_df(df):
    """
    Performs predictions and diagnostics on an entire DataFrame of students.
    Adds predicted risk, confidence, cluster, and warning flag columns.
    """
    df_copy = df.copy()
    
    # Prepare feature matrix
    X_batch = df_copy[FEATURE_COLS].copy()
    if "parental_education" in X_batch.columns:
        X_batch["parental_education"] = X_batch["parental_education"].map(EDU_MAP).fillna(1)
        
    scaled_batch = scaler.transform(X_batch)
    
    # Predictions
    rf_preds = rf_model.predict(scaled_batch)
    rf_probas = rf_model.predict_proba(scaled_batch)
    clusters = kmeans_model.predict(scaled_batch)
    
    pred_labels = [label_encoder.classes_[i] for i in rf_preds]
    confidences = [round(float(np.max(p)) * 100, 1) for p in rf_probas]
    
    cluster_map = {0: "High Performers", 1: "Moderate", 2: "At-Risk"}
    cluster_labels = [cluster_map.get(c, f"Group {c}") for c in clusters]
    
    df_copy["predicted_risk"] = pred_labels
    df_copy["confidence"] = confidences
    df_copy["cluster_segment"] = cluster_labels
    
    return df_copy
