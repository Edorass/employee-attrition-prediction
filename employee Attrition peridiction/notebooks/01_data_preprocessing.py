# 01 - Data Preprocessing

## 1. Importing Libraries

import pandas as pd
import numpy as np

## 2. Load Dataset

df = pd.read_csv(r'C:\Users\Dariush\Desktop\pythhon\ML Project\employee Attrition peridiction\data\WA_Fn-UseC_-HR-Employee-Attrition.csv')

## 3. Initial Overview

print(df.shape)
df.head

## 4. Drop Useless Columns

useless_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=useless_cols, inplace=True)

## 5. Encode Target Variable (Attrition)

df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

## 6. Check Class Imbalance

df['Attrition'].value_counts(normalize=True)

## 7. Save Cleaned Data 

df.to_csv(r'C:\Users\Dariush\Desktop\pythhon\ML Project\employee Attrition peridiction\data\cleaned_employee_data.csv', index=False)