# 02 - Exploratory Data Analysis (EDA)

## 1. Load Cleaned Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(r'C:\Users\Dariush\Desktop\pythhon\ML Project\employee Attrition peridiction\data\cleaned_employee_data.csv')
os.makedirs('../outputs/figures', exist_ok=True)

## 2. Distribution of Attrition

plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=df)
plt.title("Distribution of Attrition (Target Variable)")
plt.xticks([0, 1], ['No', 'Yes'])
plt.savefig('../outputs/figures/hist_attrition_distribution.png')
plt.show()


## 3. Numerical Features vs Attrition

num_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears']
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Attrition', y=col, data=df)
    plt.title(f'{col} vs Attrition')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig(f'../outputs/figures/box_{col}_vs_attrition.png')
    plt.show()


## 4. Categorical Features vs Attrition

cat_cols = ['BusinessTravel', 'Department', 'Gender', 'JobRole', 'OverTime']
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='Attrition', data=df)
    plt.title(f'{col} vs Attrition')
    plt.xticks(rotation=45)
    plt.savefig(f'../outputs/figures/count_{col}_vs_attrition.png')
    plt.show()


## 5. Correlation Matrix

plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig('../outputs/figures/correlation_matrix.png')
plt.show()

