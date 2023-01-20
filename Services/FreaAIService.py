from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
import itertools
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class FreaAIService():
    def __init__(self, n_datapoints_cutoff) -> None:
        self.n_datapoints_cutoff = n_datapoints_cutoff
    
    def fit_DT(self, df,predictors):
        """ Fit a classification decision tree and return key elements """

        X = df[predictors]
        y = df['ACCURACY']

        model = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)
        model.fit(X, y)

        preds = model.predict(X)
        acc = accuracy_score(y, preds)

        return model, preds, acc, X
    def return_dt_split(self, model: DecisionTreeClassifier, col, accuracy, n_datapoints_cutoff=40, col_2=None, impurity_cutoff=1.0, acc_cutoff=0.2):
        """
        Return all indices of col that meet the following criteria:
        1. Leaf has accuracy lower that baseline by acc_cutoff
        2. Split size > n_datapoints_cutoff 
        :param model: SKLearn classification decision tree model
        :param col: (pd.Series) column used to split on
        :param accuracy: (pd.Series) column corresponding to correct/incorrect classification
        :param col_2: (pd.Series) column to be used for interactions
        :param impurity_cutoff: (float) requirement for entropy/gini of leaf
        :param n_datapoints_cutoff: (int) minimum n in a final node to be returned
        :param acc_cutoff: (float) accuracy cutoff for returning float
        :return: (dict[node_idx, indices]) where indices corresponds to the col that meet the above criteria
        """

        # get leaf ids and setup
        df = pd.concat([col, col_2], axis=1) if col_2 is not None else pd.DataFrame(col)
        leaf_id = model.apply(df)
        decision = model.get_params()
        t = model.tree_
        baseline_acc = np.mean(accuracy)

        # get indices of leaf ids that meet criteria
        keeps_1 = {i for i,v in enumerate(t.n_node_samples) if v > n_datapoints_cutoff} # sample size
        keeps_2 = {i for i,v in enumerate(t.impurity) if v <= impurity_cutoff} # sample size
        keeps = keeps_1 & keeps_2

        # store all data and corresponding index
        node_indices = {}
        slice_acc = -1
        slice_acc_list=[]
        prob_idx=[]
        for idx in keeps:
            node_indices[idx] = [i for i,v in enumerate(leaf_id) if v == idx]

            # remove non-low-accuracy areas and empty lists
            slice_acc = [x[1] / sum(x) for x in t.value[idx]] 
            if baseline_acc - slice_acc < acc_cutoff or node_indices[idx] == []:
                del node_indices[idx]
                slice_acc = None
            elif baseline_acc - slice_acc >= acc_cutoff:
                slice_acc_list.append(slice_acc)
                prob_idx.append(idx)
            
        


        return (f'{col.name}{"-"+col_2.name if col_2 is not None else ""}',model,prob_idx, slice_acc_list)

    def run_data_search(self, df, n_datapoints_cutoff):
        """ Iterate over data columns and perform DT for interactions
        :param df: (pd.DataFrame) of raw data with correct/incorrect classification
        :param n_datapoints_cutoff: (int) minimum n in a final node to be returned
        """
        acc_col = df['ACCURACY']

        # store for output
        categoricals_acc = []
        bivariate_acc = []

        # univariate loop
        for col_name in [x for x in list(df) if x != 'ACCURACY']:
            c = df[col_name]
            print(f'Running: DT for {col_name}')
            predictors = [col_name]
            model, *_, X = self.fit_DT(df, predictors)


            categoricals_acc.append(self.return_dt_split(model, c, acc_col, n_datapoints_cutoff))

        for col1, col2 in itertools.combinations(set(df) - set(['ACCURACY','outcome']), 2):
                print(f'Running: DT for {col1} and {col2}')
                c1, c2 = df[col1], df[col2]
                predictors = [col1,col2]
                model, *_, X = self.fit_DT(df, predictors=predictors)
                bivariate_acc.append(self.return_dt_split(model, c1, acc_col, n_datapoints_cutoff, c2))

        return (categoricals_acc, bivariate_acc)

    def clean_output(self, b, c):
        """ 
        Take list of outputs of DT and DT interactions and return sorted 
        value by accuracy drop.
        """

        feture,model,prob_idx, slice_acc_list = [],[], [], []

        # Save vals 
        for x in b:
            if x[3]!=[]:
                feture.append(x[0])
                model.append(x[1])
                prob_idx.append(x[2])
                slice_acc_list.append(x[3])

        for x in c:
            if x[3]!=[]:
                feture.append(x[0].split("-"))
                model.append(x[1])
                prob_idx.append(x[2])
                slice_acc_list.append(x[3])

        out = pd.DataFrame(dict(feture=feture,model=model, prob_idx=prob_idx, slice_acc_list=slice_acc_list))
        out.sort_values(by=['slice_acc_list'], inplace=True)
        out.index = range(len(out.index))

        return out

    def freaAI_extract_tree_data(self, out):
        features = out.iloc[0,0]
        tree_model = out.iloc[0,1]
        leafs = out.iloc[0,2]
        self.out = out.iloc[0, :]
        return tree_model, features, leafs

    def freaAI_extract_data(self, tree_models : DecisionTreeClassifier , features, leafs, X: pd.DataFrame, y: pd.DataFrame):
        #TO DO preprocess X_train and y_train
        x_data=X.reset_index(drop=True)
        y_data=y.reset_index(drop=True)

        leaf_indices = tree_models.apply(x_data[features])
        bad_indices=np.argwhere(np.isin(leaf_indices , leafs)).flatten()
        y_bad_data = y_data.iloc[bad_indices]
        x_bad_data = x_data.iloc[bad_indices]

        y_good_data = y_data.drop(bad_indices)
        x_good_data = x_data.drop(bad_indices)
        return x_good_data, y_good_data, x_bad_data, y_bad_data

    def runFreaAI(self, processed_data):
        categoricals_acc, bivariate_acc=self.run_data_search(processed_data, self.n_datapoints_cutoff)
        out = self.clean_output(categoricals_acc, bivariate_acc)
        tree_models, features, leafs= self.freaAI_extract_tree_data(out)
        return tree_models, features, leafs
