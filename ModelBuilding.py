from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

def train_test_split_and_features(train_df, test_df):
    x_train = train_df.drop(columns=['charges'])
    y_train = train_df['charges']
    x_test = test_df.drop(columns=['charges'])
    y_test = test_df['charges']
    scaler_x = MinMaxScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    features = list(x_train.columns)
    return x_train, x_test, y_train, y_test, features

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    xgb = XGBRegressor(random_state=0)
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    xgb_r2_score = r2_score(y_test, xgb_predict)
    xgb_mean_sq_error = mean_squared_error(y_test, xgb_predict)
    xgb_mean_absolute_error = mean_absolute_percentage_error(y_test, xgb_predict)
    print("xgb_r2_score")
    print(xgb_r2_score)
    print("\n")
    print("xgb_mean_absolute_error:", xgb_mean_absolute_error * 100, '\n')
    print("xgb_mean_sq_error", xgb_mean_sq_error)
    return xgb