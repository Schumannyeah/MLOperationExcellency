# MLOperationExcellency
Machine Learning for Operation Excellency

Based on kaggle learn
Intermediate ML 
C4      -   Introduction
C402    -   Missing Values
            Q:  SimpleImputer function fit_transform and transform
            Q:  RandomForestRegressor logic

For C10 Time Series, some modules quoted are not availables from pip install.
For example the module "learntools.time_series.utils", where seasonal_plot can be imported
To solve it:
1.      go to https://github.com/Kaggle/learntools/tree/master
2.      download zip file, extract it to ‘learntools-master\learntools-master’ directly under the project root where venv can read
3.      then cmd run ‘python setup.py install', then we might run into some error like 
        "machine_learning, ml_explainability, ml-insights, ml_intermediate, python, "
        
        then go to ‘ learntools-master\learntools-master\learntools__init.py’ from the extracted folder,
        Remove ‘ml_insights, ‘ from the __init.py python file
        Save and close it.
4.      Actually after finishing the 3 steps above, it still doesn't work.
        The trick is to copy the learntools folder under the build of the zip file, then
        replace the content under venv/lib/site-packages
5.      for the kaggle_c1003_seasonality_b.py, we still needs to remove:
        "infer_datetime_format=True,", which causes future warning.
        then the codes will work.