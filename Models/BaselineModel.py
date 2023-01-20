from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from Services.DataProcessingService import DataProcessingService 
from Services.MetricService import MetricService

class BaselineModel():
    def __init__(self, params, classification_threshold) -> None:
        self.model = XGBRegressor(objective='reg:squarederror') if params == {} else XGBRegressor(**params)
        self.metricService = MetricService(classification_threshold)
        self.baseline_metrics = {"train":{}, "test":{}}
        self.X_test_baseline = None
        self.y_test_baseline = None   
        
    def run_baseline(self, df: pd.DataFrame, dataProcessingService : DataProcessingService) -> np.array: 
        df = pd.get_dummies(df,columns=dataProcessingService.get_categorical_features())
        X_train, X_test, y_train, y_test = dataProcessingService.split_data(df)
        self.model.fit(X_train, y_train)
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        self.X_test_baseline = X_test.reset_index(drop=True)
        self.y_test_baseline = y_test.reset_index(drop=True)
        self.baseline_metrics["test"] = self.metricService.calculate_metrics(y_test, y_test_pred)
        self.baseline_metrics["train"] = self.metricService.calculate_metrics(y_train, y_train_pred)
        X_train_classification = self.metricService.create_classification(y_train, y_train_pred)
        return X_train_classification

    def filter_datasets(self, x_test_index, y_test_index):
        return self.X_test_baseline.iloc[x_test_index], self.y_test_baseline.iloc[y_test_index]
        
    def predict(self, df):
        return self.model.predict(df)