#!/usr/bin/env python
# coding: utf-8

# In[5]:


from flask import Flask, request, jsonify, send_file
import zipfile
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from preprocessing.preprocessing import (
    preprocessing_pipeline,
    clean_application_train,
    clean_previous_application,
    clean_installments_payments,
    merge_bureau_and_balance,
    find_and_transform_skewed_features,
    merge_datasets,
    encode_bureau_status,
    aggregate_bureau_balance,
    feature_engineering
)


# In[ ]:


exclude_columns = [
    "TARGET",
    "SK_ID_CURR",
    "SK_ID_PREV",
    "SK_ID_BUREAU", 
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_21",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_5",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_EMAIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "NAME_CONTRACT_STATUS",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WALLSMATERIAL_MODE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WEEKDAY_APPR_PROCESS_START",
    "CODE_GENDER",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS"
]


# In[6]:


app = Flask(__name__)


# In[7]:


model = joblib.load('../models/best_model_lgbm.pkl')


# In[12]:


preprocessor = joblib.load('../models/preprocessor_lgbm.pkl')


# In[8]:


def preprocess_individual_datasets(data):
    application_train = pd.DataFrame(data['application_train'])
    bureau = pd.DataFrame(data['bureau'])
    bureau_balance = pd.DataFrame(data['bureau_balance'])
    previous_application = pd.DataFrame(data['previous_application'])
    pos_cash_balance = pd.DataFrame(data['pos_cash_balance'])
    credit_card_balance = pd.DataFrame(data['credit_card_balance'])
    installments_payments = pd.DataFrame(data['installments_payments'])
    
    application_train_cleaned = clean_application_train(application_train)
    previous_application_cleaned = clean_previous_application(previous_application)
    installments_payments_cleaned = clean_installments_payments(installments_payments)
    merged_bureau_data = merge_bureau_and_balance(bureau, bureau_balance)
    
    merged_data = merge_datasets(
        application_train_cleaned,
        previous_application_cleaned,
        installments_payments_cleaned,
        credit_card_balance,
        pos_cash_balance,
        merged_bureau_data
    )
    
    merged_data = feature_engineering(merged_data)
    merged_data = find_and_transform_skewed_features(merged_data, exclude_columns=exclude_columns)
    
    return merged_data


# In[13]:


@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and zipfile.is_zipfile(file):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                files = {name: pd.read_csv(zip_ref.open(name)) for name in zip_ref.namelist()}

                data = {
                    'application_train': files['application_train.csv'].to_dict(orient='records'),
                    'bureau': files['bureau.csv'].to_dict(orient='records'),
                    'bureau_balance': files['bureau_balance.csv'].to_dict(orient='records'),
                    'previous_application': files['previous_application.csv'].to_dict(orient='records'),
                    'pos_cash_balance': files['pos_cash_balance.csv'].to_dict(orient='records'),
                    'credit_card_balance': files['credit_card_balance.csv'].to_dict(orient='records'),
                    'installments_payments': files['installments_payments.csv'].to_dict(orient='records'),
                }

                input_data = preprocess_individual_datasets(data)

                sk_id_curr = input_data['SK_ID_CURR'].values

                input_data_processed = preprocessor.transform(input_data)

                predictions = model.predict(input_data_processed).flatten()

                output_df = pd.DataFrame({
                    'SK_ID_CURR': sk_id_curr,
                    'TARGET': predictions
                })

                output_df.to_csv('predictions.csv', index=False)

                return send_file('predictions.csv', as_attachment=True)
        else:
            return jsonify({'error': 'Uploaded file is not a valid zip file'})

    except Exception as e:
        return jsonify({'error': str(e)})


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

