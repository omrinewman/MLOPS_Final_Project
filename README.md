# MLOPS
This project contains source code and supporting files for MLOPS final project pipeline.

## Deployment

1. Copy/clone all content to a directory - you may press on the green button "Code" above then press "Download zip" and extract the content.
2. Create and activate a virtual environment by running the following commands:
```
python -m venv mlops_env 
mlops_env\Scripts\activate.bat
```
3. Install the required packages by using the provided requirements.txt file by running the following command.
```
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

## Production Model for Customer:
Please follow this section only after running the pipeline!

In order to use the pipeline for predictions on a CSV file, run the following command: (you may used the provided samples.csv file as an example (**Relevant to the French dataset**))
``` 
python predict.py samples.csv
```
## Cleanup
1. In order to remove the virtual environment, first deactivate it by running the following command and then delete the virtual environment folder.

```
deactivate
```
2. Delete the folder with the code content
## Good luck!
