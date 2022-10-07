import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# Loading Model
path = 'C:/Users/Pedro/repos/health_insurance_cross_sell/pa004_health_insurance_cross_sell/health_insurance_cross-sell/'
model = pickle.load(open(path + 'models/model_knn.pkl', 'rb'))

# Initialize API
app = Flask(__name__)


@app.route('/healthinsurance/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])

        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = HealthInsurance()

        # Data Cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # Feature Engineering
        df2 = pipeline.feature_engineering(df1)

        # Data Preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('192.168.15.59', debug=True)
