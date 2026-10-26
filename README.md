# Student Performance Prediction Platform (EduPredict AI)

**Sustainable Development Goal (SDG):** SDG 4 — Quality Education  
**Platform Domain:** Educational Data Mining (EDM) & Predictive Academic Analytics  

---

## Project Overview & Objectives

In traditional university administration, academic vulnerabilities are only exposed after end-semester examinations when failing grades and detentions are irreversible. 

The **Student Performance Prediction Platform** is an AI-driven Early Warning and Diagnostic System that monitors continuous evaluation signals (lecture attendance, internal sessional tests, class quizzes, assignment submission rates, and historical GPA) to:
1. **Classify Risk Levels**: Classify enrolled students into `High Risk`, `Medium Risk`, and `Low Risk` tiers.
2. **Early Warning Diagnostic**: Detect specific failure modes (e.g., attendance < 65%, internal tests < 25/50).
3. **Prescriptive Mentor Interventions**: Provide faculty mentors with actionable counseling protocols and remedial pathways.
4. **Unsupervised Student Grouping**: Segment students into 3 cohorts via K-Means clustering for differentiated teaching.
5. **Batch Processing**: Enable administrators to screen an entire class roster simultaneously via CSV upload.

---

## Applied Curriculum Modules

- **Unit/Module 1 (Introduction to AI & Problem Formulation):** Problem definition, educational data mining literature review, and dataset structuring.
- **Unit/Module 2 (Supervised Learning Basics):** Logistic Regression and Decision Tree baseline models, feature encoding, and standard scaling.
- **Unit/Module 3 (Unsupervised Learning & Ensemble Methods):**
  - **K-Means Clustering** with the **Elbow Method** ($k=3$) to partition students into High, Moderate, and Low performers.
  - **Random Forest Ensemble** with hyperparameter tuning (Best performing model: **95.0% Accuracy**).
  - **SMOTE (Synthetic Minority Over-sampling Technique)** to resolve class imbalance, elevating at-risk student recall from **~55% to 100%**.
  - **5-Fold Stratified Cross-Validation**, confusion matrices, precision, recall, and weighted F1-score.
  - **Feature Importance Ranking** verifying that *Internal Marks (42.1%)*, *Attendance (27.6%)*, and *Past GPA (9.9%)* are the primary drivers.
- **Unit/Module 4 (Neural Networks & Deep Learning):**
  - **Multi-Layer Perceptron (MLP)** with 2 hidden layers (64 and 32 neurons), ReLU activation, Adam optimizer, early stopping, and L2 regularization to prevent overfitting. Achieves **81.0% test accuracy**.

---

## Key Advanced Features Added

1. **Official Printable Academic Audit Report (`/report/<student_id>`)**:
   - Institutional NIET Greater Noida header, continuous assessment breakdown (attendance, internals, tests, assignments, GPA).
   - Machine Learning diagnostic classification, confidence score, early warning triggers, and formally signed Mentor Action Plan with a 1-click **Print / Save as PDF** button.
2. **Faculty Mentor Intervention Logging & History Tracking**:
   - Mentors can log counseling sessions, remedial class enrollments, and parent notifications directly into the SQLite database with timestamps and notes.
3. **Student Consultation Booking & Live Workflow**:
   - Students (like Muskan Gola) can submit consultation and doubt requests via the Student Portal. Requests appear live on Mr. Amit Yadav's dashboard with 1-click "Mark Completed" resolution.
4. **AI Hyperparameter Tuning Sandbox (`/playground`)**:
   - Interactive research sandbox allowing reviewers to adjust tree depth, estimators, and train-test splits live to observe bias-variance tradeoffs and recall sensitivities.
5. **Full Cohort Evaluated CSV Export (`/download/evaluated-cohort`)**:
   - One-click export of the entire 500-student cohort annotated with predicted risk tiers, model confidence, projected SGPA, and K-Means cluster assignments.

---

## User Roles & Database Authentication

The platform is integrated with an **SQLite database (`data/platform.db`)** with secure password hashing (`werkzeug.security`). It supports:
1. **New User Registration (`/register`)**: Allows any new user to register as either **Faculty Mentor** or **Student**.
2. **Persistent Session Tracking**: The user's **Full Name** is dynamically displayed across the navigation bar, welcome banners, and personal scorecards.
3. **Pre-configured Quick Demo Accounts**:

| Role | Username / Email | Password | Destination Dashboard | Features |
|:---|:---:|:---:|:---|:---|
| 👨‍🏫 **Faculty Mentor** | `amit` / `amit@niet.co.in` | `password123` | **[Class Executive Dashboard](http://127.0.0.1:5000/)** | Displays **"Amit Yadav"**, full cohort analytics, high-risk rosters, batch CSV uploads, intervention triggers. |
| 🎓 **Student User** | `muskan` / `muskan@niet.co.in` | `password123` | **[Student Portal](http://127.0.0.1:5000/student-dashboard)** | Displays **"Muskan Gola" (Roll: 2501330100232)**, personal scorecard, attendance safety check, interactive "What-If" simulator. |
| 🛡️ **Administrator** | `admin` / `admin@niet.co.in` | `admin123` | **[Admin Overview](http://127.0.0.1:5000/)** | Displays **"Dean Academics"**, system evaluation matrices and full student cohort data. |

---

## Model Performance Benchmarks

| Algorithm | Curriculum Unit | Accuracy | Precision | Weighted F1 | At-Risk Recall | 5-Fold CV Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Selected)** | Module 3 (Ensemble) | **95.00%** | **95.06%** | **95.02%** | **100.00%** | **97.65% ± 1.56%** |
| **Decision Tree** | Module 2 (Supervised) | 94.00% | 94.17% | 94.04% | 100.00% | 94.19% ± 2.08% |
| **Logistic Regression** | Module 2 (Baseline) | 91.00% | 91.86% | 91.18% | 88.24% | 94.81% ± 2.02% |
| **MLP Neural Network** | Module 4 (Neural Net) | 81.00% | 83.26% | 81.59% | 88.24% | 88.37% ± 2.30% |

---

## Folder Structure

```text
Student Performance Prediction Platform/
├── app.py                      # Flask Application with web routes and REST API
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows launch script
├── README.md                   # Complete PBL project documentation
├── models/
│   ├── train_models.py         # Complete training pipeline (SMOTE, RF, MLP, KMeans)
│   ├── predictor.py            # Diagnostic inference engine & early warning rules
│   └── saved/                  # Serialized trained models & metadata
│       ├── random_forest_model.joblib
│       ├── mlp_model.joblib
│       ├── decision_tree_model.joblib
│       ├── logistic_regression_model.joblib
│       ├── kmeans_model.joblib
│       ├── scaler.joblib
│       ├── label_encoder.joblib
│       └── model_metrics.json
├── data/
│   ├── generate_dataset.py     # Generates 500 anonymized student records
│   ├── students_dataset.csv    # 500 records dataset
│   └── data_dictionary.md      # Formal feature definitions & value ranges
├── static/
│   ├── css/style.css           # Modern custom styling & theme
│   └── plots/                  # Visual evidence generated from pipeline
│       ├── elbow_method.png
│       ├── kmeans_clusters.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── confusion_matrices.png
├── templates/
│   ├── base.html               # Shared layout & top navigation
│   ├── index.html              # Executive Dashboard with interactive Chart.js widgets
│   ├── students.html           # Full student cohort directory with search & filters
│   ├── predict.html            # Individual prediction calculator & diagnostic report
│   ├── batch_predict.html      # Batch CSV upload & bulk processing tool
│   ├── clustering.html         # K-Means cluster analysis & elbow method view
│   ├── models_evaluation.html  # Technical evaluation matrix & confusion matrices
│   ├── playground.html         # AI hyperparameter research playground
│   ├── report_card.html        # Printable official student academic audit report
│   ├── login.html              # Role-based sign in portal
│   ├── register.html           # New user registration portal
│   └── student_dashboard.html  # Student self-service portal & What-If simulator
└── notebooks/
    └── student_performance_analysis.ipynb # Jupyter notebook documenting EDA & ML
```

---

## Installation & Running the Prototype

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Packages: `Flask`, `scikit-learn`, `pandas`, `numpy`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`

### Steps to Run

1. **Clone or Navigate to the Directory:**
   ```bash
   cd "Student Performance Prediction Platform"
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-train Models & Regenerate Plots:**
   ```bash
   python models/train_models.py
   ```

4. **Start the Web Application:**
   - Double-click `run.bat` or run:
   ```bash
   python app.py
   ```

5. **Open in Browser:**
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Web Platform Pages & Navigation

1. **Executive Dashboard (`/`):** Cohort summary statistics, interactive Chart.js doughnut and bar charts, driving factors progress bars, and immediate attention at-risk roster.
2. **Student Directory (`/students`):** Searchable and filterable database of 500 students with color-coded risk and cluster badges, detailed view modal, and direct AI diagnosis triggers.
3. **Single Student Predictor (`/predict`):** Interactive parameter sliders, model selector (Random Forest, MLP Neural Net, etc.), quick-scenario loaders, risk classification banner, probability breakdown, cross-model consensus, early warning flags, and mentor intervention protocols.
4. **Batch Prediction (`/batch`):** Drag-and-drop CSV upload for bulk screening of an entire class roster with downloadable sample template and summarized output table.
5. **K-Means Clustering (`/clustering`):** Unsupervised student segmentation with Elbow Method curve, 2D feature scatter plot, and cluster-specific mentor guidance.
6. **Model Benchmarks (`/models`):** Academic evaluation page displaying SMOTE imbalance resolution evidence (+45% at-risk recall), confusion matrices for all algorithms, and MLP neural network specifications.
7. **AI Playground (`/playground`):** Interactive sandbox for tuning estimators, tree depth, and neural network parameters live.
8. **Student Portal (`/student-dashboard`):** Self-service scorecard with real-time "What-If" academic performance simulator.
9. **Printable Academic Audit (`/report/<student_id>`):** Certified individual academic audit and early warning diagnosis report.
