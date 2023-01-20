import numpy as np
class ClassificationService():
    def __init__(self, classification_threshold) -> None:
        self.classification_threshold = classification_threshold

    def create_classification(self, target, pred) -> np.array:
        return np.where(abs(pred - target) < self.classification_threshold + self.classification_threshold*target, True, False)
