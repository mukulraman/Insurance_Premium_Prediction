import pandas as pd
import joblib

from DataCleaning import data_cleaning

from ModelBuilding import(
    train_test_split_and_features,
    fit_and_evaluate_model
)    

# Read the Heart Disease Training Data from output_data.csv file in data folder
# If output_data.csv file in data folder doesn't exist, then first run the DB_to_CSV.py script
df = pd.read_csv('data/output_data.csv')

df=data_cleaning(df)

train_df = df.iloc[:700000]
test_df = df.iloc[700000:900000]

x_train, x_test, y_train, y_test, features = train_test_split_and_features(train_df, test_df)
xgb_model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)

joblib.dump(xgb_model, 'models\Model_Classifier_Premium.pkl')
joblib.dump(features, 'models\Features_Columns.pkl')