"""
Student Performance Prediction Platform - Flask Web Application
AI-Driven Academic Early Warning & Performance Diagnostic System
SDG 4: Quality Education
"""

import os
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, flash, session

from models.predictor import (
    predict_single, predict_batch_df, MODEL_METRICS, FEATURE_COLS
)
from models.database import (
    init_db, create_user, authenticate_user,
    log_intervention, get_interventions,
    submit_counseling_request, get_counseling_requests, update_counseling_status
)

app = Flask(__name__)
app.secret_key = "niet-group63-student-performance-secret-key"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "students_dataset.csv")

# Ensure database is initialized
init_db()

@app.context_processor
def inject_user():
    """Provides current user object to all templates."""
    user = session.get("user")
    if not user:
        # Default session as Amit Yadav (Faculty Mentor)
        user = {
            "name": "Amit Yadav",
            "role": "Faculty Mentor",
            "department": "Computer Science & Engineering",
            "email": "amit@niet.co.in",
            "student_id": "",
            "avatar": "AY"
        }
    return {"current_user": user}

def get_dataset():
    """Loads current student dataset from CSV."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

@app.route("/")
def index():
    """Dashboard / Class Summary view."""
    df = get_dataset()
    total_students = len(df)
    
    if total_students > 0:
        high_risk_count = int((df["risk_level"] == "High Risk").sum())
        med_risk_count = int((df["risk_level"] == "Medium Risk").sum())
        low_risk_count = int((df["risk_level"] == "Low Risk").sum())
        
        avg_attendance = round(float(df["attendance"].mean()), 1)
        avg_internal = round(float(df["internal_marks"].mean()), 1)
        avg_gpa = round(float(df["past_gpa"].mean()), 2)
        avg_study = round(float(df["study_hours"].mean()), 1)
        
        # Recent high-risk students needing intervention
        critical_students = df[df["risk_level"] == "High Risk"].head(6).to_dict(orient="records")
    else:
        high_risk_count = med_risk_count = low_risk_count = 0
        avg_attendance = avg_internal = avg_gpa = avg_study = 0.0
        critical_students = []

    return render_template(
        "index.html",
        total_students=total_students,
        high_risk_count=high_risk_count,
        med_risk_count=med_risk_count,
        low_risk_count=low_risk_count,
        high_risk_pct=round((high_risk_count / total_students * 100) if total_students else 0, 1),
        avg_attendance=avg_attendance,
        avg_internal=avg_internal,
        avg_gpa=avg_gpa,
        avg_study=avg_study,
        critical_students=critical_students,
        interventions=get_interventions(limit=6),
        counseling_requests=get_counseling_requests(mentor_name="Amit Yadav"),
        metrics=MODEL_METRICS
    )

@app.route("/students")
def students():
    """Student Directory with filtering, searching, and sorting."""
    df = get_dataset()
    risk_filter = request.args.get("risk", "All")
    cluster_filter = request.args.get("cluster", "All")
    search_query = request.args.get("q", "").strip().lower()
    
    filtered_df = df.copy()
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]
        
    if cluster_filter != "All":
        filtered_df = filtered_df[filtered_df["cluster_label"].str.contains(cluster_filter, case=False, na=False)]
        
    if search_query:
        filtered_df = filtered_df[
            filtered_df["name"].str.lower().str.contains(search_query) |
            filtered_df["student_id"].str.lower().str.contains(search_query)
        ]
        
    student_list = filtered_df.to_dict(orient="records")
    return render_template(
        "students.html",
        students=student_list,
        total_count=len(student_list),
        current_risk=risk_filter,
        current_cluster=cluster_filter,
        search_query=search_query
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Individual Student Prediction Calculator & Early Warning Diagnostic."""
    result = None
    if request.method == "POST":
        input_data = {
            "attendance": request.form.get("attendance", 75.0),
            "internal_marks": request.form.get("internal_marks", 35.0),
            "assignment_rate": request.form.get("assignment_rate", 80.0),
            "study_hours": request.form.get("study_hours", 14.0),
            "class_test_marks": request.form.get("class_test_marks", 22.0),
            "past_gpa": request.form.get("past_gpa", 7.5),
            "parental_education": request.form.get("parental_education", "Bachelor"),
            "extracurricular": request.form.get("extracurricular", 0),
            "internet_access": request.form.get("internet_access", 1)
        }
        chosen_model = request.form.get("model_choice", "Random Forest")
        result = predict_single(input_data, model_choice=chosen_model)
        
    return render_template("predict.html", result=result, metrics=MODEL_METRICS)

@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    """Batch CSV Prediction upload and analysis."""
    batch_results = None
    summary = None
    
    if request.method == "POST":
        if "csv_file" not in request.files:
            flash("Please upload a CSV file.", "danger")
            return redirect(request.url)
            
        file = request.files["csv_file"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)
            
        try:
            df = pd.read_csv(file)
            # Check required feature columns
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                flash(f"Uploaded CSV is missing columns: {', '.join(missing)}", "danger")
                return redirect(request.url)
                
            processed_df = predict_batch_df(df)
            batch_results = processed_df.head(50).to_dict(orient="records")
            summary = {
                "total": len(processed_df),
                "high_risk": int((processed_df["predicted_risk"] == "High Risk").sum()),
                "med_risk": int((processed_df["predicted_risk"] == "Medium Risk").sum()),
                "low_risk": int((processed_df["predicted_risk"] == "Low Risk").sum())
            }
        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "danger")
            
    return render_template("batch_predict.html", batch_results=batch_results, summary=summary)

@app.route("/clustering")
def clustering():
    """Module 3: Unsupervised Learning & K-Means Clustering Insights."""
    return render_template("clustering.html", metrics=MODEL_METRICS)

@app.route("/models")
def models_evaluation():
    """Module 3 & 4 Model Comparison, SMOTE Impact, and Technical Evaluation."""
    return render_template("models_evaluation.html", metrics=MODEL_METRICS)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Role-based authentication portal using SQLite database."""
    if request.method == "POST":
        identifier = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        
        user = authenticate_user(identifier, password)
        if user:
            session["user"] = user
            flash(f"Welcome, {user['name']}! You have successfully signed in as {user['role']}.", "success")
            if user["role"] == "Student":
                return redirect(url_for("student_dashboard"))
            return redirect(url_for("index"))
        else:
            flash("Invalid email/username or password. Please try again or use the 1-click demo logins.", "danger")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Sign up portal for new faculty members and students."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        role = request.form.get("role", "Student").strip()
        department = request.form.get("department", "Computer Science & Engineering").strip()
        student_id = request.form.get("student_id", "").strip()
        
        if password != confirm_password:
            flash("Passwords do not match. Please re-enter.", "danger")
            return redirect(url_for("register"))
            
        if len(password) < 4:
            flash("Password must be at least 4 characters long.", "danger")
            return redirect(url_for("register"))
            
        user, err = create_user(
            name=name,
            email=email,
            password=password,
            role=role,
            department=department,
            student_id=student_id if role == "Student" else ""
        )
        
        if err:
            flash(f"Registration failed: {err}", "danger")
            return redirect(url_for("register"))
            
        # Log the newly registered user in immediately
        session["user"] = user
        flash(f"Account successfully created! Welcome, {user['name']}.", "success")
        if user["role"] == "Student":
            return redirect(url_for("student_dashboard"))
        return redirect(url_for("index"))
        
    return render_template("register.html")

@app.route("/logout")
def logout():
    """Clears user session."""
    session.pop("user", None)
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/student-dashboard")
def student_dashboard():
    """Dedicated Student Self-Service Dashboard & What-If Simulator."""
    user = session.get("user")
    if not user:
        # Default to Muskan Gola if student view visited without prior login
        user = {
            "name": "Muskan Gola",
            "role": "Student",
            "department": "BTech CSE-D",
            "student_id": "2501330100232",
            "email": "muskan@niet.co.in",
            "avatar": "MG"
        }
    return render_template("student_dashboard.html", metrics=MODEL_METRICS, user=user)

# --- REST API Endpoints ---

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API for single prediction."""
    data = request.get_json(force=True)
    model_choice = data.get("model_choice", "Random Forest")
    res = predict_single(data, model_choice=model_choice)
    return jsonify(res)

@app.route("/api/stats")
def api_stats():
    """REST API for dashboard statistics and charts."""
    df = get_dataset()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
        
    risk_dist = df["risk_level"].value_counts().to_dict()
    cluster_dist = df["cluster_label"].value_counts().to_dict()
    
    return jsonify({
        "total_students": len(df),
        "risk_distribution": risk_dist,
        "cluster_distribution": cluster_dist,
        "feature_importance": MODEL_METRICS.get("feature_importance", {}),
        "model_benchmarks": {
            name: {
                "accuracy": m["accuracy"],
                "f1_score": m["f1_score"],
                "high_risk_recall": m["high_risk_recall"]
            }
            for name, m in MODEL_METRICS.get("models", {}).items()
        }
    })

@app.route("/report/<student_id>")
def report_card(student_id):
    """Generates official printable academic audit & early warning diagnosis."""
    df = get_dataset()
    match = df[df["student_id"] == student_id]
    if match.empty:
        flash(f"Student ID {student_id} not found.", "warning")
        return redirect(url_for("students"))
        
    student = match.iloc[0].to_dict()
    pred = predict_single(student)
    return render_template("report_card.html", student=student, pred=pred, metrics=MODEL_METRICS)

@app.route("/interventions/log", methods=["POST"])
def add_intervention():
    """Logs an intervention action by faculty mentor in the database."""
    student_id = request.form.get("student_id", "").strip()
    student_name = request.form.get("student_name", "").strip()
    action_type = request.form.get("action_type", "1-on-1 Counseling Session").strip()
    notes = request.form.get("notes", "").strip()
    
    current_u = session.get("user", {})
    mentor_name = current_u.get("name", "Amit Yadav")
    
    log_intervention(student_id, student_name, mentor_name, action_type, notes)
    flash(f"Intervention recorded for {student_name} ({action_type})", "success")
    return redirect(request.referrer or url_for("index"))

@app.route("/counseling/request", methods=["POST"])
def request_counseling():
    """Student submits a consultation/doubt clearing request."""
    user = session.get("user", {})
    name = user.get("name", request.form.get("student_name", "Muskan Gola"))
    email = user.get("email", request.form.get("student_email", "muskan@niet.co.in"))
    student_id = user.get("student_id", "2501330100232")
    topic = request.form.get("topic", "Academic Guidance & Attendance Improvement")
    message = request.form.get("message", "")
    
    submit_counseling_request(name, email, student_id, "Amit Yadav", topic, message)
    flash("Your consultation request has been submitted to Faculty Mentor Mr. Amit Yadav.", "success")
    return redirect(url_for("student_dashboard"))

@app.route("/counseling/resolve/<int:req_id>", methods=["POST"])
def resolve_counseling(req_id):
    """Mentor marks a consultation request completed."""
    update_counseling_status(req_id, "Completed")
    flash("Consultation request marked as completed.", "success")
    return redirect(url_for("index"))

@app.route("/playground")
def playground():
    """Interactive hyperparameter tuning and model research sandbox."""
    return render_template("playground.html", metrics=MODEL_METRICS)

@app.route("/download/evaluated-cohort")
def download_evaluated_cohort():
    """Exports full 500-student cohort with predicted risks, clusters, and confidence scores."""
    df = get_dataset()
    evaluated_df = predict_batch_df(df)
    output = io.BytesIO()
    evaluated_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="NIET_evaluated_student_cohort_500.csv"
    )

@app.route("/download/sample-template")
def download_sample_template():
    """Downloads sample CSV template for batch prediction."""
    df = get_dataset()
    sample_df = df[FEATURE_COLS].head(10)
    output = io.BytesIO()
    sample_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_prediction_sample_template.csv"
    )

if __name__ == "__main__":
    # Production / Cloud deployment port binding
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
