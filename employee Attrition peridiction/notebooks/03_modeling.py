# 03 - Modeling & Evaluation

## 1. Setup & Load Clean Data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE   # pip install imblearn

df = pd.read_csv('../data/cleaned_employee_data.csv')

## 2. One-Hot/Label Encode Categorical Features

target = 'Attrition'
y = df[target]
X = df.drop(columns=[target])

cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns


## 3. Train-Test Split

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

## 4. Handle Class Imbalance

from imblearn.pipeline import Pipeline as ImbPipeline

smote = SMOTE(random_state=42)

## 5. Scale Numerical Features

log_reg = ImbPipeline(steps=[
    ('prep', preprocess),
    ('smote', smote),
    ('model', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))

## 6. Baseline Models

rf = ImbPipeline(steps=[
    ('prep', preprocess),
    ('smote', smote),
    ('model', RandomForestClassifier(
        n_estimators=300, max_depth=None,
        class_weight='balanced', random_state=42))
])

rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

## 7. Ensemble / Tuned Model

import os

os.makedirs('../outputs/models', exist_ok=True)

import joblib
joblib.dump(rf, '../outputs/models/employee_attrition_rf.joblib')
