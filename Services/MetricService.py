from sklearn import metrics
import numpy as np
class MetricService():
    def __init__(self, classification_threshold) -> None:
        self.classification_threshold = classification_threshold

    def create_classification(self, target, pred) -> np.array:
        return np.where(abs(pred - target) < self.classification_threshold + self.classification_threshold*target, True, False)

    def calculate_metrics(self, target, y_pred):
        MAE = metrics.mean_absolute_error(target, y_pred)
        MSE = metrics.mean_squared_error(target, y_pred)
        ACCURACY = self.create_classification(target, y_pred).mean()
        return {'MAE': MAE , 'MSE': MSE, 'ACCURACY': ACCURACY}
    
    def print_metrics(self, metrics: dict):
        for metric in metrics.keys():
            print(f'{metric}: {metrics[metric]: .2f}')