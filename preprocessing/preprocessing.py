#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[4]:


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
    "CNT_FAM_MEMBERS",
]

def find_and_transform_skewed_features(
    df, skewness_threshold=1, exclude_columns=None, shift_constant=1
):
    """
    Finds highly skewed numerical features and applies logarithmic transformation.

    parameters:
    df: data frame containing numerical features.
    skewness_threshold: threshold for skewness coef.
    exclude_columns: list of columns to exclude from the transformation.
    shift_constant: this is the constant to add to features to handle zeros and negative values.

    returns:
    Dataframe with log-transformed variables.
    """
    if exclude_columns is None:
        exclude_columns = []

    df_to_transform = df.drop(columns=exclude_columns, errors="ignore")

    skewness = df_to_transform.skew(numeric_only=True).sort_values(ascending=False)

    highly_skewed_features = skewness[abs(skewness) > skewness_threshold].index.tolist()
    
    for feature in highly_skewed_features:
        min_value = df_to_transform[feature].min()
        if min_value <= 0:
            df_to_transform[feature] = np.log1p(
                df_to_transform[feature] - min_value + shift_constant
            )
        else:
            df_to_transform[feature] = np.log1p(df_to_transform[feature])

    df.update(df_to_transform)

    print(f"Log-transformed features: {highly_skewed_features}")

    return df

def feature_engineering(df):
    df.replace(365243, np.nan, inplace=True)
    
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 0.00001)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 0.00001)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 0.00001)
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 0.00001)
    
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / -365
    df['REGISTRATION_YEARS'] = df['DAYS_REGISTRATION'] / -365
    df['ID_PUBLISH_YEARS'] = df['DAYS_ID_PUBLISH'] / -365

    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_FAM_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    
    # 4. External Sources and Aggregates
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_MEDIAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)
    
    # 5. Flag Counts
    df['FLAG_DOCUMENT_SUM'] = df[[f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]].sum(axis=1)
    df['FLAG_PHONE_SUM'] = df[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE']].sum(axis=1)

    return df

def clean_application_train(df):
    df = df[df["CODE_GENDER"] != "XNA"]
    
    features_to_convert = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "REGION_RATING_CLIENT",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "REGION_RATING_CLIENT_W_CITY",
    ]

    for feature in features_to_convert:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    return df

def clean_previous_application(df):
    features_to_remove = [
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED",
        "AMT_DOWN_PAYMENT",
    ]
    df = df.drop(columns=features_to_remove, errors="ignore")

    df = df[df["AMT_ANNUITY"] >= 0]

    days_features = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    bin_edges = [-np.inf, -1500, -1000, -500, 0, 99999]  
    bin_labels = [
        "Very Old",
        "Old",
        "Recent",
        "Very Recent",
        "Missing",
    ]  
    
    for feature in days_features:
        df[f"{feature}_BINNED"] = pd.cut(
            df[feature].replace(
                365243, -99999
            ), 
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=False,
        )
    df.drop(columns=days_features, inplace=True)

    return df

def clean_installments_payments(df):
    df["PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]

    df = df[df["AMT_PAYMENT"] >= 0]

    return df

def clean_bureau(df):
    features_to_remove = ["AMT_ANNUITY", "AMT_CREDIT_MAX_OVERDUE"]

    df = df.drop(columns=features_to_remove, errors="ignore")

    features_to_convert = ["CNT_CREDIT_PROLONG"]

    for feature in features_to_convert:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    return df

def encode_bureau_status(bureau_balance):
    status_mapping = {
        "C": 0,  # Closed
        "0": 1,  # No DPD
        "1": 2,  # DPD 1-30
        "2": 3,  # DPD 31-60
        "3": 4,  # DPD 61-90
        "4": 5,  # DPD 91-120
        "5": 6,  # DPD 120+ 
        "X": 7,  # Unknown status 
    }

    bureau_balance["STATUS_ENCODED"] = bureau_balance["STATUS"].map(status_mapping)

    return bureau_balance

def aggregate_bureau_balance(bureau_balance):
    bureau_balance_agg = (
        bureau_balance.groupby("SK_ID_BUREAU")
        .agg(
            MONTHS_BALANCE_MEAN=("MONTHS_BALANCE", "mean"),
            STATUS_MEAN=("STATUS_ENCODED", "mean"),
            STATUS_MAX=("STATUS_ENCODED", "max"),
            STATUS_MIN=("STATUS_ENCODED", "min"),
        )
        .reset_index()
    )

    bureau_balance_agg.columns = ["SK_ID_BUREAU"] + [
        f"BUREAU_BAL_{col.upper()}" for col in bureau_balance_agg.columns[1:]
    ]

    return bureau_balance_agg

def merge_bureau_and_balance(bureau_df, bureau_balance_df):
    bureau_cleaned = clean_bureau(bureau_df)
    
    bureau_cleaned_transformed = find_and_transform_skewed_features(bureau_cleaned)

    bureau_balance_encoded = encode_bureau_status(bureau_balance_df)

    bureau_balance_agg = aggregate_bureau_balance(bureau_balance_encoded)

    merged_bureau_data = bureau_cleaned.merge(
        bureau_balance_agg, on="SK_ID_BUREAU", how="left"
    )

    return merged_bureau_data

def merge_datasets(app_df, prev_df, inst_df, card_df, pos_cash_df, merged_bureau_data):
    prev_app_agg = (
        prev_df.groupby("SK_ID_CURR")
        .agg(
            {
                "SK_ID_PREV": "nunique",  # unique previous applications
                "AMT_ANNUITY": "mean",  #mean of the annuity amount
                "AMT_APPLICATION": "mean",  #mean of the application amount
                "AMT_CREDIT": "mean",  #mean of the credit amount
                "AMT_GOODS_PRICE": "mean",  #mean of the goods price
                "DAYS_FIRST_DRAWING_BINNED": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
                "DAYS_FIRST_DUE_BINNED": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
                "DAYS_LAST_DUE_1ST_VERSION_BINNED": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
                "DAYS_LAST_DUE_BINNED": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
                "DAYS_TERMINATION_BINNED": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
            }
        )
        .reset_index()
    )

    prev_app_agg.columns = ["SK_ID_CURR", "PREV_APP_COUNT"] + [
        f"PREV_{col.upper()}_{'MODE' if col.endswith('_BINNED') else 'MEAN'}"
        for col in prev_app_agg.columns[2:]
    ]

    inst_pay_agg = (
        inst_df.groupby("SK_ID_CURR")
        .agg(
            {
                "NUM_INSTALMENT_NUMBER": "nunique", 
                "DAYS_INSTALMENT": "mean", 
                "DAYS_ENTRY_PAYMENT": "mean", 
                "AMT_INSTALMENT": "mean",
                "AMT_PAYMENT": "mean",
            }
        )
        .reset_index()
    )

    inst_pay_agg.columns = ["SK_ID_CURR", "INST_COUNT"] + [
        f"INST_{col.upper()}_MEAN" for col in inst_pay_agg.columns[2:]
    ]

    credit_card_agg = (
        card_df.groupby("SK_ID_CURR")
        .agg(
            {
                "SK_ID_PREV": "nunique",
                "MONTHS_BALANCE": "mean",
                "AMT_BALANCE": "mean",
                "AMT_CREDIT_LIMIT_ACTUAL": "mean",
                "AMT_DRAWINGS_ATM_CURRENT": "mean",
                "AMT_DRAWINGS_CURRENT": "mean",
                "AMT_DRAWINGS_OTHER_CURRENT": "mean",
                "AMT_DRAWINGS_POS_CURRENT": "mean",
                "AMT_INST_MIN_REGULARITY": "mean",
                "AMT_PAYMENT_TOTAL_CURRENT": "mean",
                "AMT_RECEIVABLE_PRINCIPAL": "mean",
                "AMT_RECIVABLE": "mean",
                "AMT_TOTAL_RECEIVABLE": "mean",
                "CNT_DRAWINGS_ATM_CURRENT": "mean",
                "CNT_DRAWINGS_CURRENT": "mean",
                "CNT_DRAWINGS_OTHER_CURRENT": "mean",
                "CNT_DRAWINGS_POS_CURRENT": "mean",
                "CNT_INSTALMENT_MATURE_CUM": "mean",
                "NAME_CONTRACT_STATUS": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
                "SK_DPD": "mean",  
                "SK_DPD_DEF": "mean",  
            }
        )
        .reset_index()
    )


    credit_card_agg.columns = ["SK_ID_CURR", "CREDIT_PREV_LOANS_COUNT"] + [
        f"CREDIT_{col.upper()}_MEAN"
        if col != "NAME_CONTRACT_STATUS"
        else f"CREDIT_{col.upper()}_MODE"
        for col in credit_card_agg.columns[2:]
    ]

    pos_cash_agg = (
        pos_cash_df.groupby("SK_ID_CURR")
        .agg(
            {
                "SK_ID_PREV": "nunique",
                "MONTHS_BALANCE": "mean",
                "CNT_INSTALMENT": "mean",
                "CNT_INSTALMENT_FUTURE": "mean",
                "NAME_CONTRACT_STATUS": lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else None,
                "SK_DPD": "mean",
                "SK_DPD_DEF": "mean",
            }
        )
        .reset_index()
    )

    pos_cash_agg.columns = ["SK_ID_CURR", "POS_PREV_LOANS_COUNT"] + [
        f"POS_{col.upper()}_MEAN"
        if col != "NAME_CONTRACT_STATUS"
        else f"POS_{col.upper()}_MODE"
        for col in pos_cash_agg.columns[2:]
    ]

    bureau_final_agg = (
        merged_bureau_data.groupby("SK_ID_CURR")
        .agg(
            {
                "SK_ID_BUREAU": "nunique",
                "CREDIT_ACTIVE": lambda x: (x == "Active").sum(),
                "CREDIT_CURRENCY": "nunique",
                "DAYS_CREDIT": "mean",
                "CREDIT_DAY_OVERDUE": "mean",
                "DAYS_ENDDATE_FACT": "mean",
                "CNT_CREDIT_PROLONG": "sum",
                "AMT_CREDIT_SUM": "mean",
                "AMT_CREDIT_SUM_DEBT": "mean",
                "AMT_CREDIT_SUM_LIMIT": "mean",
                "AMT_CREDIT_SUM_OVERDUE": "mean",
                "CREDIT_TYPE": "nunique",
                "DAYS_CREDIT_UPDATE": "mean",
                "BUREAU_BAL_MONTHS_BALANCE_MEAN": "mean",
                "BUREAU_BAL_STATUS_MEAN": "mean",
                "BUREAU_BAL_STATUS_MAX": "max",
                "BUREAU_BAL_STATUS_MIN": "min",
            }
        )
        .reset_index()
    )

    bureau_final_agg.columns = ["SK_ID_CURR"] + [
        f"BUREAU_{col.upper()}" for col in bureau_final_agg.columns[1:]
    ]

    merged_data = app_df.merge(prev_app_agg, on="SK_ID_CURR", how="left")
    merged_data = merged_data.merge(inst_pay_agg, on="SK_ID_CURR", how="left")
    merged_data = merged_data.merge(credit_card_agg, on="SK_ID_CURR", how="left")
    merged_data = merged_data.merge(pos_cash_agg, on="SK_ID_CURR", how="left")
    merged_data = merged_data.merge(bureau_final_agg, on="SK_ID_CURR", how="left")

    return merged_data


# In[3]:


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, filter_column=None, filter_value=None):
        self.columns_to_drop = columns_to_drop
        self.filter_column = filter_column
        self.filter_value = filter_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.columns_to_drop:
            X = X.drop(columns=self.columns_to_drop, errors="ignore")
        
        if self.filter_column and self.filter_value is not None:
            X = X[X[self.filter_column] != self.filter_value]
        
        return X


# In[2]:


def preprocessing_pipeline(X):
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    if "SK_ID_CURR" in numerical_features:
        numerical_features.remove("SK_ID_CURR")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # Impute missing values with 0
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = Pipeline(
        steps=[
            ("custom", CustomTransformer(
                columns_to_drop=["SK_ID_CURR"], 
                filter_column="CODE_GENDER", 
                filter_value="XNA" 
            )),
            ("num_cat", ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_features),
                    ("cat", categorical_pipeline, categorical_features),
                ]
            ))
        ]
    )

    return preprocessor


# In[ ]:


def get_feature_names(preprocessor, numerical_features, categorical_features):
    numeric_feature_names = numerical_features   
    cat_transformer = preprocessor.named_transformers_['cat']
    categorical_feature_names = cat_transformer.get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)
    
    return all_feature_names

