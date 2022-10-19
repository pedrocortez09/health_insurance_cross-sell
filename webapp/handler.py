import pickle
import pandas as pd
import os
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# Loading Model
model = pickle.load(open('models/model_lgbm_tuned.pkl', 'rb'))

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
    port = os.environ.get( 'PORT', 5000 )
    app.run(host='0.0.0.0', port=port)
