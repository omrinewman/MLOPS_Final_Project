import sys
import pickle
import numpy as np
from xgboost import XGBRegressor
import warnings
import pickle
from Services.FreaAIService import FreaAIService
from Services.DataProcessingService import DataProcessingService 
from Services.MetricService import MetricService
from Models.BaselineModel import BaselineModel
from Services.GraphService import GraphService
from Services.DBLoadService import DbLoader
from config import PARAMS




class Pipline():
  def __init__(self, df, target_column, split_ratio, params, n_datapoints_cutoff, threshold):
    self.freaAI_service = FreaAIService(n_datapoints_cutoff)
    self.dataProcessingService = DataProcessingService(df, target_column, split_ratio)
    self.baseline_model = BaselineModel(params, threshold)
    self.metric_service = MetricService(threshold)
    self.df = df
    self.params = params
    self.X_train, self.X_test, self.y_train, self.y_test = self.dataProcessingService.split_data(df)

  def run_pipeline(self):
    print("Starting the Pipeline process!")
    print("Running baseline model and create binary Classification")
    X_train_classification= self.baseline_model.run_baseline(self.df, self.dataProcessingService)
    print("Printing baseline model result metrics on test:")
    self.metric_service.print_metrics(self.baseline_model.baseline_metrics["test"])

    print("Pre Processing Data for the new model")
    self.X_train = self.dataProcessingService.run_pre_process_pipeline(self.X_train, 'train')
    self.X_test = self.dataProcessingService.run_pre_process_pipeline(self.X_test,'test')
    self.X_train["ACCURACY"] = X_train_classification

    print("#####################################")
    print("Running FreaAI....")
    tree_model, features, leafs= self.freaAI_service.runFreaAI(self.X_train)
    self.X_train.drop(columns=["ACCURACY"], inplace = True)
    print("FreaAI analysis is over!")

    print("#####################################")
    print("Extracting low and high quality data segments based on FreaAI Decision Trees")
    x_train_good_data, y_train_good_data, x_train_bad_data, y_train_bad_data = self.freaAI_service.freaAI_extract_data(tree_model, features, leafs, self.X_train , self.y_train)
    x_test_good_data, y_test_good_data, x_test_bad_data, y_test_bad_data = self.freaAI_service.freaAI_extract_data(tree_model, features, leafs, self.X_test, self.y_test)

    print("#####################################") 
    print("Calibrate new XGBOOST model based on the low-quality data segments")
    _, pred_on_bad_data, bad_data_metrics =  self.calibrate_and_run_new_model(x_train_bad_data, y_train_bad_data, x_test_bad_data, y_test_bad_data, "low-quality")
   
    print("#####################################")
    print("Calibrate new XGBOOST model based on the high-quality data segments")
    high_quality_model, pred_on_good_data, good_data_metrics =  self.calibrate_and_run_new_model(x_train_good_data, y_train_good_data, x_test_good_data, y_test_good_data, "high-quality")
  
    total_predicts = np.concatenate((pred_on_bad_data, pred_on_good_data), axis=0)
    full_y_test = np.concatenate((y_test_bad_data, y_test_good_data), axis=0)
    combined_metrics = self.metric_service.calculate_metrics(full_y_test, total_predicts)
    print("Printing results metrics on all of test data combined:")
    self.metric_service.print_metrics(combined_metrics)

    print("#####################################")
    print("Printing results metrics on the full X_test data based on the new high-quality_data_based model:")
    pred_on_X_test = high_quality_model.predict(self.X_test)
    X_test_data_metrics = self.metric_service.calculate_metrics(self.y_test, pred_on_X_test)
    self.metric_service.print_metrics(X_test_data_metrics)
    print("#####################################")
    baseline_metrics_on_bad_data = self.run_basline_on_data_segments(x_test_bad_data, y_test_bad_data, bad_data_metrics, "low-quality")
    print("#####################################")
    baseline_metrics_on_good_data = self.run_basline_on_data_segments(x_test_good_data, y_test_good_data, good_data_metrics, "high-quality")
    
    with open('good_model.pkl', 'wb') as model_file:
      pickle.dump(high_quality_model, model_file)
  
    with open('pre_processor.pkl', 'wb') as model_file:
      pickle.dump(self.dataProcessingService, model_file) 

    print('Saving results graphs to Pics directory:')
    graphService = GraphService()
    graphService.create_graph(baseline_metrics_on_bad_data, bad_data_metrics, 'on Low-quality data', "Low_quality")
    graphService.create_graph(baseline_metrics_on_good_data, good_data_metrics, 'on High-quality data', "High_quality")
    graphService.create_graph(self.baseline_model.baseline_metrics["test"], combined_metrics, 'on full test data', "Combined_results")
    graphService.create_graph(self.baseline_model.baseline_metrics["test"], X_test_data_metrics, 'on full test data - Good model', "Full_test_good_model")

  def run_basline_on_data_segments(self, x_test, y_test, metrics, datatype):
      print(f"Predicting on {datatype} test data segments using the baseline model")
      x_test_for_baseline, y_test_for_baseline = self.baseline_model.filter_datasets(x_test.index, y_test.index)
      baseline_pred = self.baseline_model.predict(x_test_for_baseline)
      baseline_metrics = self.metric_service.calculate_metrics(y_test_for_baseline, baseline_pred)
      print(f"Printing result metrics on {datatype} test data segments using the baseline model")
      improved_model_acc = metrics['ACCURACY']
      baseline_model_acc = baseline_metrics['ACCURACY']
      print(f"printing Accuracy of baseline vs {datatype}_data_based model on {datatype} data")
      print(f'Accuracy of baseline model: {baseline_model_acc: .2f}')
      print(f'Accuracy of {datatype}_data_based model: {improved_model_acc: .2f}')
      print(f'Percentage improvement: {(improved_model_acc/ baseline_model_acc -1)*100: .2f}%')
      return baseline_metrics

  def calibrate_and_run_new_model(self, x_train, y_train, x_test, y_test, datatype):
      model = self.calibrate_model(x_train, y_train)
      print(f"Predicting on {datatype} test data segments using the new {datatype}_data_based XGBOOST model")
      pred = model.predict(x_test)
      metrics = self.metric_service.calculate_metrics(y_test, pred)
      print(f"Printing results metrics on {datatype} classified test data:")
      self.metric_service.print_metrics(metrics)
      return model, pred, metrics
    
  def calibrate_model(self, X, y):
    model = XGBRegressor(objective='reg:squarederror') if self.params == {} else XGBRegressor(**self.params)
    model.fit(X, y)
    return model

if __name__ == '__main__':
  warnings.filterwarnings('ignore')
  data_type = sys.argv[1]
  dbloader = DbLoader()
  if data_type in dbloader.dataset_dict.keys():
    dataset = dbloader.dataset_dict[data_type]()
    model = Pipline(dataset, **PARAMS[data_type])
    model.run_pipeline()
  else:
    print("This Pipeline version currently only supports the French and Boston dataset")