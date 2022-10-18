import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# Loading Model
path = 'C:/Users/Pedro/repos/health_insurance_cross_sell/pa004_health_insurance_cross_sell/health_insurance_cross-sell/'
model = pickle.load(open(path + 'models/model_lgbm_tuned.pkl', 'rb'))

# Initialize API
app = Flask(__name__)


@app.route('/healthinsurance/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):  # unique row
            test_raw = pd.DataFrame(test_json, index=[0])

        else:  # multiple rows
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        test_raw_copy = test_raw.copy()

        # instantiate HealthInsurance class
        pipeline = HealthInsurance()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        print('apos df1')
        print(test_raw.head())
        print(test_raw.head().values)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        print('apos df2')
        print(test_raw.head())
        print(test_raw.head().values)

        # data preparation
        df3 = pipeline.data_preparation(df2)
        print('apos df3')
        print(test_raw.head())
        print(test_raw.head().values)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw_copy, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('192.168.15.59', debug=True)
