<h1 align="center">🔍 Employee Attrition Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-RandomForest-green" />
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

<p align="center">
  Predicting whether an employee will leave the company using supervised machine learning techniques.
</p>

---

## 📌 Project Overview

This project focuses on predicting **employee attrition** (whether an employee will quit or stay) using various HR-related features like age, job role, income, and overtime status.

🎯 Target: `Attrition` (Yes / No)

---

## 🧠 Algorithms Used

- ✅ Logistic Regression
- ✅ Random Forest (Best performing)
- ✅ Decision Tree
- ✅ K-Nearest Neighbors

---

## 📊 Exploratory Data Analysis

Here are some key visuals from the EDA phase:

### 📈 Attrition Distribution

<img src="outputs/figures/hist_Distribution of Attrition (Target Variable).png" width="450"/>

### 🧬 Correlation Matrix

<img src="outputs/figures/hist_Correlation Matrix.png" width="550"/>

---

## 🧪 Model Performance

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 84.5%    | 0.73     |
| Random Forest      | **87.6%**| **0.79** |
| Decision Tree      | 83.2%    | 0.71     |

✅ **Random Forest** had the best results in balancing precision and recall.

---

## 📁 Project Structure

employee-attrition-prediction/
│
├── data/ # Cleaned dataset
├── notebooks/ # Jupyter notebooks for EDA, modeling
├── outputs/
│ ├── figures/ # Sample EDA plots
│ └── models/ # Trained model (not pushed)
├── requirements.txt # Install dependencies
├── README.md # You're here!
└── .gitignore
---

## 🛠️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Edorass/employee-attrition-prediction.git
cd employee-attrition-prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks
cd notebooks
jupyter notebook


🌍 Multi-Lingual Summary | خلاصه فارسی
این پروژه با استفاده از الگوریتم‌های یادگیری ماشین سعی می‌کند پیش‌بینی کند که آیا کارمند از شرکت استعفا خواهد داد یا خیر. تمرکز بر تحلیل ویژگی‌هایی مانند سن، سمت شغلی، درآمد، سابقه کاری و اضافه‌کاری است.

🙋 About Me
👤 Developed by Dariush
📧 Reach out: LinkedIn
🔗 GitHub Profile

⭐ Star This Repo
If you found this helpful, please give it a ⭐ to support the project!

