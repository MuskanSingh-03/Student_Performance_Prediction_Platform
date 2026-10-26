"""
Model Training & Evaluation Pipeline for Student Performance Prediction Platform
Applies Module 3 (Ensemble Methods, K-Means Clustering, SMOTE) and Module 4 (Neural Networks - MLP).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "students_dataset.csv")
    models_dir = os.path.join(base_dir, "models", "saved")
    plots_dir = os.path.join(base_dir, "static", "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Dataset
    print("[1/6] Loading student dataset...")
    df = pd.read_csv(data_path)
    
    # Feature columns
    feature_cols = [
        "attendance", "internal_marks", "assignment_rate", "study_hours",
        "class_test_marks", "past_gpa", "parental_education",
        "extracurricular", "internet_access"
    ]
    
    X = df[feature_cols].copy()
    y = df["risk_level"].copy()
    
    # Categorical encoding
    edu_mapping = {"High School": 0, "Bachelor": 1, "Master": 2}
    X["parental_education"] = X["parental_education"].map(edu_mapping).fillna(1)
    
    # Target encoding: 'High Risk': 0, 'Medium Risk': 1, 'Low Risk': 2
    label_encoder = LabelEncoder()
    # Ensure consistent order: High Risk, Medium Risk, Low Risk
    label_encoder.classes_ = np.array(["High Risk", "Medium Risk", "Low Risk"])
    y_encoded = label_encoder.transform(y)
    
    # Train / Test split (80:20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Imbalance Handling with SMOTE
    print("[2/6] Applying SMOTE for class balancing...")
    # Baseline check before SMOTE with Random Forest
    rf_imbalanced = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_imbalanced.fit(X_train, y_train)
    y_pred_imb = rf_imbalanced.predict(X_test)
    recall_before_smote = float(recall_score(y_test, y_pred_imb, labels=[0], average=None)[0])
    
    # SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # 3. Model Training (Module 1, 2, 3, 4)
    print("[3/6] Training Machine Learning & Neural Network models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=9, min_samples_split=4, random_state=42),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=0.005, max_iter=500, early_stopping=True, random_state=42
        )
    }
    
    trained_models = {}
    metrics = {
        "smote_improvement": {
            "at_risk_recall_before": round(recall_before_smote * 100, 2),
            "at_risk_recall_after": 0.0
        },
        "models": {},
        "feature_importance": {},
        "clusters": {}
    }
    
    confusion_matrices = {}
    
    for name, model in models.items():
        # Train on SMOTE balanced data
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted'))
        rec = float(recall_score(y_test, y_pred, average='weighted'))
        f1 = float(f1_score(y_test, y_pred, average='weighted'))
        
        # Per-class report
        class_rec = recall_score(y_test, y_pred, labels=[0, 1, 2], average=None)
        high_risk_recall = float(class_rec[0])
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
        confusion_matrices[name] = cm
        
        # 5-Fold Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
        
        metrics["models"][name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "high_risk_recall": round(high_risk_recall * 100, 2),
            "cv_mean_accuracy": round(float(cv_scores.mean()) * 100, 2),
            "cv_std": round(float(cv_scores.std()) * 100, 2),
            "confusion_matrix": cm
        }
        
        trained_models[name] = model
        print(f" -> {name}: Accuracy={acc*100:.2f}%, F1={f1*100:.2f}%, High-Risk Recall={high_risk_recall*100:.2f}%")
        
    metrics["smote_improvement"]["at_risk_recall_after"] = metrics["models"]["Random Forest"]["high_risk_recall"]
    
    # 4. Feature Importance (Random Forest)
    print("[4/6] Extracting feature importance...")
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    metrics["feature_importance"] = {col: round(float(val) * 100, 2) for col, val in feat_series.items()}
    
    # 5. Unsupervised Learning: K-Means Clustering & Elbow Method (Module 3)
    print("[5/6] Executing K-Means clustering and elbow method...")
    X_full_scaled = scaler.transform(X)
    
    inertias = []
    k_range = list(range(2, 9))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_full_scaled)
        inertias.append(float(km.inertia_))
        
    # Fit final K-Means with K=3 (Low, Medium, High performers)
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_3.fit_predict(X_full_scaled)
    df["cluster"] = cluster_labels
    
    # Determine cluster meanings based on avg attendance & internal marks
    cluster_stats = df.groupby("cluster")[["attendance", "internal_marks", "past_gpa"]].mean()
    sorted_clusters = cluster_stats.sort_values(by="attendance", ascending=False).index.tolist()
    
    # Map cluster ids to human labels
    cluster_names_map = {
        sorted_clusters[0]: "High Performers (Consistent)",
        sorted_clusters[1]: "Moderate Performers (Developing)",
        sorted_clusters[2]: "At-Risk / Low Performers"
    }
    df["cluster_label"] = df["cluster"].map(cluster_names_map)
    
    # Save cluster profile
    for c_id, label in cluster_names_map.items():
        subset = df[df["cluster"] == c_id]
        metrics["clusters"][label] = {
            "count": int(len(subset)),
            "avg_attendance": round(float(subset["attendance"].mean()), 1),
            "avg_internal_marks": round(float(subset["internal_marks"].mean()), 1),
            "avg_past_gpa": round(float(subset["past_gpa"].mean()), 2),
            "avg_study_hours": round(float(subset["study_hours"].mean()), 1)
        }
        
    # Update dataset with cluster info
    df.to_csv(data_path, index=False)
    
    # 6. Generate Evaluation Plots
    print("[6/6] Generating evaluation plots & saving models...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Plot 1: Elbow Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_range, inertias, marker='o', color='#2563eb', linewidth=2.5, markersize=8)
    plt.axvline(x=3, color='#dc2626', linestyle='--', label='Optimal Elbow Point (k=3)')
    plt.title("K-Means Elbow Method for Optimal Clusters (Module 3)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Clusters (k)", fontsize=11)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "elbow_method.png"), dpi=200)
    plt.close()
    
    # Plot 2: K-Means Clusters Scatter
    plt.figure(figsize=(8, 5))
    palette = {"High Performers (Consistent)": "#16a34a", "Moderate Performers (Developing)": "#d97706", "At-Risk / Low Performers": "#dc2626"}
    sns.scatterplot(
        data=df, x="attendance", y="internal_marks", hue="cluster_label",
        palette=palette, style="cluster_label", s=80, alpha=0.85
    )
    plt.title("Student Segmentation via K-Means Clustering", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Attendance Percentage (%)", fontsize=11)
    plt.ylabel("Internal Marks (out of 50)", fontsize=11)
    plt.legend(title="Student Segment", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kmeans_clusters.png"), dpi=200)
    plt.close()
    
    # Plot 3: Feature Importance
    plt.figure(figsize=(8, 5))
    feat_df = pd.DataFrame({"Feature": feat_series.index, "Importance (%)": feat_series.values * 100})
    sns.barplot(data=feat_df, x="Importance (%)", y="Feature", hue="Feature", legend=False, palette="Blues_r")
    plt.title("Feature Importance Ranking (Random Forest)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Importance Score (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=200)
    plt.close()
    
    # Plot 4: Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    model_names = list(metrics["models"].keys())
    accuracies = [metrics["models"][m]["accuracy"] for m in model_names]
    recalls = [metrics["models"][m]["high_risk_recall"] for m in model_names]
    f1s = [metrics["models"][m]["f1_score"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.bar(x - width, accuracies, width, label='Overall Accuracy (%)', color='#3b82f6')
    plt.bar(x, f1s, width, label='Weighted F1-Score (%)', color='#10b981')
    plt.bar(x + width, recalls, width, label='At-Risk Recall (%)', color='#f59e0b')
    
    plt.xlabel('Algorithm', fontsize=11, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
    plt.title('Performance Comparison Across ML & Neural Network Models', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, model_names, fontsize=10)
    plt.ylim(60, 100)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=200)
    plt.close()
    
    # Plot 5: Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    classes = ["High Risk", "Medium", "Low Risk"]
    
    for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=classes, yticklabels=classes, cbar=False)
        axes[idx].set_title(f"{m_name} (Acc: {metrics['models'][m_name]['accuracy']}%)", fontweight='bold')
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        
    plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=200)
    plt.close()
    
    # Save Models, Scaler, and Metrics
    joblib.dump(trained_models["Random Forest"], os.path.join(models_dir, "random_forest_model.joblib"))
    joblib.dump(trained_models["MLP Neural Network"], os.path.join(models_dir, "mlp_model.joblib"))
    joblib.dump(trained_models["Decision Tree"], os.path.join(models_dir, "decision_tree_model.joblib"))
    joblib.dump(trained_models["Logistic Regression"], os.path.join(models_dir, "logistic_regression_model.joblib"))
    joblib.dump(kmeans_3, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.joblib"))
    
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"All models, scaler, metrics, and plots saved successfully!")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
