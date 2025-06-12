# Employee Attrition Prediction 🎯

This project aims to predict whether an employee will leave the company, using HR data and machine learning.

## 📁 Project Structure

- `data/`: Raw and cleaned datasets.
- `notebooks/`: Step-by-step scripts for preprocessing, EDA, and modeling.
- `outputs/`: Trained model and visualizations.

## 🛠️ Tech Stack

- Python
- Pandas, Seaborn, Matplotlib
- Scikit-learn
- Joblib

## ✅ Project Steps

1. **Data Cleaning:** Handled missing values and encoded categorical features.
2. **EDA:** Visualized relationships and key drivers of attrition.
3. **Modeling:** Trained a Random Forest classifier with 87% accuracy.

## 🔍 Key Insights

- OverTime and Job Role are major predictors.
- Employees with less total working years tend to leave more.

## 📈 Final Model

Saved under: `outputs/models/employee_attrition_rf.joblib`

## 🚀 How to Run

```bash
pip install -r requirements.txt
cd notebooks
python 01_data_preprocessing.py
python 02_eda.py
python 03_modeling.py

👨‍💻 Author
Dariush Ajorloo – Data Science Challenge Submission

