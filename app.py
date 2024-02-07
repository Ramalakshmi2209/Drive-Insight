from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
from joblib import load
app = Flask(__name__)
xgboost_model = load('xgboost_model.joblib')
random_forest_model = load('random_forest.joblib')
train = pd.read_csv('Training_Dataset.csv')
train.rename({'available_hours': 'online_hours'}, axis=1, inplace=True)
train = train[['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age', 'number_of_kids', 'online_hours']]
train['dayofweek'] = train['dayofweek'].astype('int64')
train['age'] = train['age'].astype('int64')
train['number_of_kids'] = train['number_of_kids'].astype('int64')

def confidence(predictions):
   
    prediction_std = predictions.std()

    
    confidence_threshold = 0.5
    
    
    if prediction_std < confidence_threshold:
        return True
    else:
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            driver_id = int(request.form['driver_id'])
            
            
            if driver_id not in train['driver_id'].unique():
                return render_template('result.html', result=f"Driver ID {driver_id} does not exist in the dataset.")

            date = request.form['date']

            input_data = pd.DataFrame({'date': [date], 'driver_id': [driver_id]})
            input_data['date'] = pd.to_datetime(input_data['date'])
            input_data['dayofweek'] = input_data['date'].dt.dayofweek
            input_data['weekend'] = input_data['dayofweek'].apply(lambda x: 0 if x < 5 else 1)

            driver_profile = pd.read_csv('driver.csv')
            input_data = pd.merge(input_data, driver_profile, on=['driver_id'])
           


            input_data['gender'].replace({'FEMALE': 1, 'MALE': 0}, inplace=True)
            input_data['online_hours'] = -1

            test_data = pd.concat([train, input_data])

            test_data['date'] = pd.to_datetime(test_data['date'])
            test_data = test_data.set_index(
                ['date', 'driver_id']
            ).unstack().fillna(method='ffill').asfreq(
                'D'
            ).stack().sort_index(level=1).reset_index()

            test_data['dayofweek'].fillna(test_data['date'].dt.dayofweek, inplace=True)
            test_data['weekend'] = test_data['dayofweek'].apply(lambda x: 0 if x < 5 else 1)

            test_data = test_data.sort_values(by=['driver_id', 'date']).drop_duplicates(subset=['date', 'driver_id'])
            test_data = test_data.set_index(['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age', 'number_of_kids'])

            test_data['lag_1'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(1)
            test_data['lag_2'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(2)
            test_data['lag_3'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(3)
            test_data['lag_4'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(4)
            test_data['lag_5'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(5)
            test_data['lag_6'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(6)
            test_data['lag_7'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(7)

            test_data['online_hours'] = pd.to_numeric(test_data['online_hours'], errors='coerce')
            test_data['online_hours'].dropna(inplace=True)

            test_data['rolling_mean'] = test_data.groupby('driver_id')['online_hours'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()).shift(1)

            test_data = test_data.reset_index(drop=False).dropna()
            test_data = test_data[['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age',
                                   'number_of_kids', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
                                   'lag_5', 'lag_6', 'lag_7', 'rolling_mean', 'online_hours']]

            def reset_test(test_data):
                test_data = test_data.set_index(['date', 'driver_id', 'dayofweek', 'weekend', 'gender', 'age', 'number_of_kids'])
                test_data['lag_1'] = test_data.groupby(level=['driver_id'])['online_hours'].shift(1)
                test_data['lag_2'] = test_data.groupby(level=['driver_id'])['lag_1'].shift(1)
                test_data['lag_3'] = test_data.groupby(level=['driver_id'])['lag_2'].shift(1)
                test_data['lag_4'] = test_data.groupby(level=['driver_id'])['lag_3'].shift(1)
                test_data['lag_5'] = test_data.groupby(level=['driver_id'])['lag_4'].shift(1)
                test_data['lag_6'] = test_data.groupby(level=['driver_id'])['lag_5'].shift(1)
                test_data['lag_7'] = test_data.groupby(level=['driver_id'])['lag_6'].shift(1)
                test_data = test_data.reset_index()
                test_data['rolling_mean'] = test_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7']].mean(
                    axis=1)
                return test_data

            chunk = test_data[
                (test_data['date'] == datetime.strptime(date, '%Y-%m-%d')) & (test_data['driver_id'] == driver_id)]
            X = chunk.iloc[:, 2:-1]
            
            y_xgboost = xgboost_model.predict(X)
            y_xgboost = round(y_xgboost[0], 1)
            
            y_random_forest = random_forest_model.predict(X)
            y_random_forest = round(y_random_forest[0], 1)

            if confidence(y_xgboost) > confidence(y_random_forest):
                result_value = y_xgboost
                result_model = 'XGBoost'
            else:
                result_value = y_random_forest
                result_model = 'Random Forest'

            test_data.loc[
                (test_data['date'] == datetime.strptime(date, '%Y-%m-%d')) & (test_data['driver_id'] == driver_id),
                'online_hours'] = result_value

            test_data = reset_test(test_data)

            result = test_data[['date', 'driver_id', 'online_hours']]
            result['date'] = pd.to_datetime(result['date'])

            threshold = 5  
            demand_status = "In Demand" if result_value > threshold else "Not In Demand"

            result_html = f" {driver_id} on {date}: {result_value} hours<br>Model Used: {result_model}<br> {demand_status}"
            return render_template('result.html', result=result_html)
        except Exception as e:
            return render_template('result.html', result=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
