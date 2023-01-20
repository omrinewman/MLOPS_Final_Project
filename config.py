french_split_ratio = {'train':0.70, 'validation':0.15, 'handout':0.15}
french_baseline_hyperparams = {'eval_metric':'poisson-nloglik', 'objective': 'count:poisson','colsample_bytree':0.9, \
  'learning_rate':0.1, 'max_depth':3, 'min_child_weight':1, 'reg_alpha':5,'subsample':0.9 }
boston_split_ratio = {'train':0.6, 'validation':0.4, 'handout':0} 
boston_baseline_hyperparams = {}
french_params =  {"target_column": "Freq", "split_ratio":french_split_ratio, "params":french_baseline_hyperparams, "n_datapoints_cutoff":2500, "threshold":0.18}
boston_params = {"target_column": "PRICE", "split_ratio":boston_split_ratio, "params":boston_baseline_hyperparams, "n_datapoints_cutoff":40, "threshold":0.07}
PARAMS = {"French":french_params, "Boston": boston_params}
  