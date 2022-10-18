import pickle
import pandas as pd


class HealthInsurance(object):

    def __init__(self):
        self.homepath = 'C:/Users/Pedro/repos/health_insurance_cross_sell/pa004_health_insurance_cross_sell/health_insurance_cross-sell/'
        self.age_scaler = pickle.load(open(self.homepath + 'src/features/age_scaler.pkl', 'rb'))
        self.annual_premium_scaler = pickle.load(open(self.homepath + 'src/features/annual_premium_scaler.pkl', 'rb'))
        self.policy_sales_channel_scaler = pickle.load( open(self.homepath + 'src/features/fe_policy_sales_channel_scaler.pkl', 'rb'))
        self.gender_scaler = pickle.load(open(self.homepath + 'src/features/target_encode_gender_scaler.pkl', 'rb'))
        self.region_code_scaler = pickle.load(open(self.homepath + 'src/features/target_encode_region_code_scaler.pkl', 'rb'))
        self.vintage_scaler = pickle.load(open(self.homepath + 'src/features/vintage_scaler.pkl', 'rb'))

    def data_cleaning(self, data):
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code',
                    'previously_insured', 'vehicle_age', 'vehicle_damage',
                    'annual_premium', 'policy_sales_channel', 'vintage']

        data.columns = cols_new

        return data

    def feature_engineering(self, data):
        # vehicle_Age
        data['vehicle_age'] = data['vehicle_age'].apply(
            lambda x: 'below_1_year' if x == '< 1 Year' else 'between_1_2_year' if x == '1-2 Year' else 'over_2_years')

        # vehicle_damage
        data['vehicle_damage'] = data['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)

        return data

    def data_preparation(self, data):
        # Annual Premium
        data['annual_premium'] = self.annual_premium_scaler.transform(data[['annual_premium']].values)

        # Age
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # Vintage
        data['vintage'] = self.vintage_scaler.transform(data[['vintage']].values)

        # Gender
        data.loc[:, 'gender'] = data['gender'].map(self.gender_scaler)

        # Region Code
        data.loc[:, 'region_code'] = data['region_code'].map(self.region_code_scaler)

        # Vehicle Age
        data = pd.get_dummies(data, prefix='vehicle_age', columns=['vehicle_age'])

        # Policy Sales Channel - Target Encoding / Frequency Encoding
        data.loc[:, 'policy_sales_channel'] = data['policy_sales_channel'].map(self.policy_sales_channel_scaler)

        # Best Features
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel',
                         'previously_insured']

        return data[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # model prediciton
        pred = model.predict_proba(test_data)

        # creating a prediction dataframe of the predict_proba (class 0 and 1 predictions)
        table_proba = pd.DataFrame(pred)

        # join prediction into original data
        original_data['Score'] = table_proba[1]

        original_data.sort_values('Score', ascending=False, inplace=True)

        return original_data.to_json(orient='records', date_format='iso')