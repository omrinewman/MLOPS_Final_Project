import sys
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from Services.DataProcessingService import DataProcessingService

def find_num_cat_features(df):
    categorical_features = df.select_dtypes(include=['object']).columns.to_list()
    numeric_features = df.columns.drop(categorical_features).to_list()
    return categorical_features, numeric_features

def create_pre_process_pipeline(categorical_features, numeric_features):
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
          transformers=[
            ('categorical', OrdinalEncoder(), categorical_features)], remainder='passthrough')
    return preprocessor

def run_pre_process_pipeline(preprocessor, df, categorical_features, numeric_features):
    processed_data = preprocessor.transform(df)
    return pd.DataFrame(processed_data, columns=categorical_features +numeric_features)

with open('good_model.pkl', 'rb') as fid:
    model = pickle.load(fid)

with open('pre_processor.pkl', 'rb') as fid:
    preprocessor = pickle.load(fid)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)
print("Preprocessing of the samples:")
processed_df = preprocessor.run_pre_process_pipeline(df,"test")

pred = model.predict(processed_df).round(0)
df= pd.concat([df, pd.DataFrame(pred, columns=['Freq'])], axis =1)
print('Saving prediction to predictions.csv file')
df.to_csv("predictions.csv")
print("Printing predictions of first 10 samples:\n")
print(df.head(10))



