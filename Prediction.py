import pandas as pd
import joblib
from DataCleaning import data_cleaning

xgb_model = joblib.load('models\Model_Classifier_Premium.pkl')
features = joblib.load('models\Features_Columns.pkl')

df = pd.read_csv('data/output_data.csv')

live_df = df.iloc[900000:]
live_df = data_cleaning(live_df)

target_column = 'charges'

if target_column in live_df.columns:
    live_df = live_df.drop(columns=[target_column])
live_df_encoded = pd.get_dummies(live_df)
for col in features:
    if col not in live_df_encoded.columns:
        live_df_encoded[col] = 0

live_df_encoded = live_df_encoded[features]

predictions = xgb_model.predict(live_df_encoded)

print("Predictions:", predictions)

prediction_series = pd.Series(predictions, name='predicted_charges')
live_df_with_predictions = live_df.reset_index(drop=True).join(prediction_series)

print(live_df_with_predictions)