import pandas as pd

def data_cleaning(df):

   df.replace([''], pd.NA, inplace=True)
   df['age'] = pd.to_numeric(df['age'], errors='coerce')
   df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
   df['children'] = pd.to_numeric(df['children'], errors='coerce')
   df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
 
   Not_Nan=['bmi','age','children']

   for mean_column in Not_Nan:
      mean_value = df[mean_column].mean()
      df[mean_column].fillna(mean_value, inplace=True)

   Not_Null=['gender','medical_history','family_medical_history','occupation']

   for column in Not_Null:
      mode_value = df[column].mode()[0]
      df[column].fillna(mode_value, inplace=True)

   df['gender'] = df['gender'].map({'male': 1, 'female': 0})
   df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    
   df = pd.get_dummies(df, columns=['region', 'medical_history', 'family_medical_history', 'exercise_frequency','occupation','coverage_level'], dtype=int)

   return df