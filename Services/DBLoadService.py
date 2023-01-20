import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

class DbLoader(): 
    def __init__(self) -> None:
        self.dataset_dict = {"French": self.load_french_db, "Boston": self.load_boston_db} 
    def load_french_db(self):
        french_df = pd.read_csv("https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff",
                 quotechar="'")
        french_df.rename(lambda x: x.replace('"', ''), axis='columns', inplace=True)
        french_df['Freq']=french_df['ClaimNb']/french_df['Exposure']
        french_df.drop(columns=['IDpol','ClaimNb','Exposure'], inplace=True)
        return french_df

    def load_boston_db(self):
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        boston_df = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
            'PTRATIO', 'B', 'LSTAT'])
        boston_df['PRICE'] = target 
        return boston_df