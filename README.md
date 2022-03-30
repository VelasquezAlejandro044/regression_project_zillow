# regression_project_zillow
Zillo Data Regression Project
By: Alejandro Velasquez 

# Estimating Home Value
----------------------

## Main Goal 
- We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.



## Project Objectives

Construct a model to predict assessed home value for single family properties using regression techniques.

Ask exploratory questions of your data that will help you understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.

Make recommendations to a data science team about how to improve predictions.

Record and share: work done, why, goals, findings, your methodologies, and your conclusions.

Create modules taht will make the proscess repeateable and the report easier to read and follow

Document code, process, findings, and key takeaways in a Jupyter Notebook Final Report.

----------------------
# Initia Questions 

- Does the size of the house affect `tax_assessed_value_target` ---> x = square_feet
- Does the amount of bathrooms and bedrooms affect 
`tax_assessed_value_target` --> x = rooms_&_bathrooms (feature engeniering by adding rooms and bathrooms)
- Are the square feet related to the tax extimated value?
- Does zipcode affect `tax_assessed_value_target` ---> x = zip_code
---------------------------------------------------------------

## Acquire Data

Use file `acquire.py` that will upload data to the final noteboolk.

--------------

idx  |Feature                           |Not null values |data type|
| --- | ---------------------------------|----------------|--------|  
| 0   |bedrooms                       | 50611 non-null  | float64  |
| 1   |bathrooms          | 50611 non-null  | float64  |
| 2   |square_feet                  | 50611 non-null  | float64  |
| 3   |year                    | 50611 non-null  | float64  |
| 4   |tax_of_property                           | 50611 non-null  | float64  |
| 5   |tax_assessed_value_target                        | 50611 non-null  | float64  |
| 6   |fips                            | 50611 non-null  | float64  |
| 7   |zip_code                     | 50611 non-null  | float64  |
| 8   |y_hat                    | 50611 non-null  | float64 |
| 9   |yhat_baseline                   | 50611 non-null  | float64|
| 10  |residual                         | 50611 non-null  | float64|
| 11  |residual_baseline                 | 50611 non-null  | float64|
datetime64[ns]
