# MLOPS
This project contains source code and supporting files for MLOPS final project pipeline.

## Deployment

1. Please use Python 3.8.10
2. Copy/clone all content to a directory - you may press on the green button "Code" and press "Download zip"
3. In the directory, create a virtual environment using the provided requirements.txt file, by running the following lines:

```
python -m venv mlops_env 
mlops_env\Scripts\activate.bat
pip install -r requirements.txt
```

## Running the Pipeline:

In order to run the pipeline for the French dataset, run the following command:
```
python pipeline.py "French"
```
For the Boston dataset, run the following command:
```
python pipeline.py "Boston"
```
During the pipeline's run, the results will be shown on screen, the final results graphs will be saves in the 'Pics' directory

## Using the Model for Prediction:
In order to use the pipeline for predictions on a CSV file, run the following command: (you may used the provided samples.csv file as an example (Relevant to the French dataset)
``` 
python predict.py samples.csv
```

## Good luck!
