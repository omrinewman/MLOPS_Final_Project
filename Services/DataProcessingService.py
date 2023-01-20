from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessingService():
    def __init__(self, df: pd.DataFrame, target_column: str, split_ratio: float) -> None:
        self.target_column = target_column
        self.split_ratio = split_ratio
        self.categorical_features, self.numeric_features = self.find_num_cat_features(df)
        self.pipeline = self.create_pre_process_pipeline()
    
    def find_num_cat_features(self, df):
        categorical_features = df.select_dtypes(include=['object']).columns.to_list()
        numeric_features = df.columns.drop(categorical_features).to_list()
        numeric_features.remove(self.target_column)
        return categorical_features, numeric_features

    def split_data(self, df: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(df.drop([self.target_column], axis = 1), \
        df[self.target_column], train_size = self.split_ratio['train'], test_size = self.split_ratio['validation'], random_state = 10)
        return X_train, X_test, y_train, y_test

    def create_pre_process_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', OrdinalEncoder(), self.categorical_features)], remainder='passthrough')
        return preprocessor

    def run_pre_process_pipeline(self, df: pd.DataFrame, mode: str):
        if mode == 'train':
            processed_data = self.pipeline.fit_transform(df)
        else: 
            processed_data = self.pipeline.transform(df)
        return pd.DataFrame(processed_data, columns=self.categorical_features + self.numeric_features)

    def get_categorical_features(self):
        return self.categorical_features

    def get_numeric_features(self):
        return self.numeric_features
