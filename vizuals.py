#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#pipeline imports

#stats imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr
#Hypothesis alpha
alpha = .05

# --------------------------------- Show distribution ----------------------------------

def explore_vizuals(df):
    # Creates boxplot fro all xs
    #We don't want to plot the `year` and `zip_code` columns.
    plt.figure(figsize=(10,7))

    # Create boxplots for all
    sns.boxplot(data=df.drop(columns=['year', 'zip_code', 'fips']))
    plt.show()

    # Ditribution of data
    df.hist(figsize=(24, 10), bins=20)


    plt.figure(figsize=(16, 10))

    # List of columns
    cols = df.columns.values

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(2,4, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

    # List of columns
    cols = ['bedrooms', 'bathrooms', 'square_feet', 'year', 'tax_of_property', 'tax_assessed_value_target', 'zip_code']



    # loop for boxlots 
    plt.figure(figsize=(20, 10))

    for i, col in enumerate(cols):

        # start count at 
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(2,4, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        plt.boxplot(df[col])

def base_line_prediction_viz(pred_mean, pred_median, y_train):
    plt.vlines(ymin = 0, ymax = 25000, x = pred_mean, color='red', alpha=.5)
    plt.vlines(ymin = 0, ymax = 25000, x = pred_median, color='orange', alpha = .5)

    plt.hist(y_train.tax_assessed_value_target, color='blue', alpha=.5, label="Actual Tax Value")
    plt.hist(y_train.baseline_pred_mean, bins=20, color='red', alpha=.5, rwidth=100, label="Predicted Tax Value - Mean")
    plt.hist(y_train.baseline_pred_median, bins=20, color='orange', alpha=.5, rwidth=100, label="Predicted Tax Value - Median")

    plt.title('Baseline Prediction ')
    plt.xlabel('Tax Value')
    plt.ylabel('Number of Houses')
    plt.legend()

def validate_scatter_plot(y_validate, pred_mean):
    # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.plot(y_validate.tax_assessed_value_target, y_validate.baseline_pred_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, (pred_mean + 1000)))
    plt.plot(y_validate.tax_assessed_value_target, y_validate.tax_assessed_value_target, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (30000, 0), rotation=25)

    plt.scatter(y_validate.tax_assessed_value_target, y_validate.taxvalue_pred_lm, 
                alpha=.5, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_assessed_value_target, y_validate.taxvalue_pred_glm, 
                alpha=.5, color="yellow", s=100, label="Model: TweedieRegressor GLM")
    plt.scatter(y_validate.tax_assessed_value_target, y_validate.taxvalue_pred_lars, 
                alpha=.5, color="blue", s=100, label="Model LassoLars")
    plt.legend()
    plt.show()

def residual_scatter_plot(y_validate, pred_mean):
    # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_validate.tax_assessed_value_target, y_validate.taxvalue_pred_lm-y_validate.tax_assessed_value_target, 
                alpha=.5, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_assessed_value_target, y_validate.taxvalue_pred_glm-y_validate.tax_assessed_value_target, 
                alpha=.5, color="black", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_assessed_value_target, y_validate.taxvalue_pred_lars-y_validate.tax_assessed_value_target, 
                alpha=.5, color="blue", s=100, label="Lars Model")
    plt.legend()
    plt.xlabel("Actual Final Grade")
    plt.ylabel("Residual/Error: Predicted Grade - Actual Grade")
    plt.title("Do the size of errors change as the actual value changes?")
    plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()

def validate_histogram(y_validate):
    plt.figure(figsize=(10,10))
    plt.hist(y_validate.tax_assessed_value_target, color = 'blue',alpha = 0.5, label = 'Actual Tax Value')
    plt.hist(y_validate.taxvalue_pred_lm, color = 'red',alpha = 0.5, label = 'Model: Linear Regression')
    plt.hist(y_validate.taxvalue_pred_glm, color = 'yellow',alpha = 0.5, label = 'Model : Tweedie Regressor')
    plt.hist(y_validate.taxvalue_pred_lars, color = 'green',alpha=0.5,label='Model: LassoLars')
    
    plt.legend();    
    
def test_histogram(y_test):
    plt.figure(figsize=(10,10))
    plt.hist(y_test.tax_assessed_value_target, color = 'blue',alpha = 0.5, label = 'Actual Tax Value')
    plt.hist(y_test.taxvalue_pred_lm, color = 'red',alpha = 0.5, label = 'Model: Linear Regression')
    plt.hist(y_test.taxvalue_pred_lars, color = 'green',alpha=0.5,label='Model: LassoLars')
    
    plt.legend();      

    
    
    
    
    
    
    