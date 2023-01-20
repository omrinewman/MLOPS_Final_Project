import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import os

class GraphService():
    def __init__(self) -> None:
        if not os.path.exists("./Pics"):
            os.makedirs("./Pics")
    def create_graph(self, baseline_metrics, model_mestrics, title, file_name):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))  

        df = pd.DataFrame(data={'Model': ["base", "pipeline"], 'Metric': [round(baseline_metrics['ACCURACY']*100,2), round(model_mestrics['ACCURACY']*100,2)]})
        sns.barplot(data=df, x='Model', y='Metric', ax= ax[0])
        ax[0].set_title("Accuracy " + title)
        ax[0].set_ylabel("Accuracy",fontsize = 12)
        h_tick = int(df['Metric'].max())
        range = np.linspace(0, int(h_tick*1.2), 5)
        ax[0].set_yticks(range)
        for i in ax[0].containers:
            ax[0].bar_label(i,)

        df = pd.DataFrame(data={'Model': ["base", "pipeline"], 'Metric': [round(baseline_metrics['MSE'],2), round(model_mestrics['MSE'],2)]})
        sns.barplot(data=df, x='Model', y='Metric', ax= ax[1])
        ax[1].set_title("MSE " + title)
        ax[1].set_ylabel("MSE",fontsize = 12)
        h_tick = int(df['Metric'].max())
        ax[1].set_yticks(np.linspace(0, h_tick*1.2, 5))
        for i in ax[1].containers:
            ax[1].bar_label(i,) 

        fig.tight_layout()
        plt.savefig(f"./Pics/{file_name}.jpg")
