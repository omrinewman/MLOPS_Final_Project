import sys
import pandas as pd
import pickle


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



