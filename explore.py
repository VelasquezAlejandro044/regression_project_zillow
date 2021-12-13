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

# --------------------------------- Show Pairplot ----------------------------------
def pairplot_explore(train):
    sns.pairplot(train, corner=True)
    plt.suptitle("sns.pairplot visualizes continuous variable relationships")
    plt.show()

# ------------------------------------ Bivariant Exploration -----------------------------------------

def bivariate_categorical(target, categorical_feature, train):
    """
    Takes in a target and plots it against categorial variables. 
    Outputs boxplots and barplots and the mean of the target.
    """
    for feature in categorical_feature:
        print(f"{feature} vs {target}")

        sns.boxplot(x=feature, y=target, data=train)
        plt.show()

        print()

        sns.barplot(x=feature, y=target, data=train)
        plt.show()
        
        print("-------------------------------")
        print(f"Mean {target} by {feature}:  ")
        print(train.groupby(feature)[target].mean())
        print()

# ------------------------------------ Stat Test -----------------------------------------

#  1. Do houses with more `bedrooms` have a higher `tax_assessed_value_target`?

def question_1(train, alpha):
    # Plot vizual 
    sns.barplot(x = 'bedrooms',y='tax_assessed_value_target',data=train, palette='pastel')

    # pandas crosstab to make a 'contingency' table
    observe = pd.crosstab(train.tax_assessed_value_target, train.bedrooms)
    chi2, p, degf, expected = stats.chi2_contingency(observe)
    print('Observed\n')
    print(observe.values)
    print('---\nExpected\n')
    print(expected.astype(int))
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < alpha:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis/")

# 2. Do houses with more `bedrooms` have a higher `tax_assessed_value_target`?

def question_2(train, alpha):
    # Plot vizual 
    sns.barplot(x = 'bathrooms',y='tax_assessed_value_target',data=train, palette='pastel', capsize=.2)

    # pandas crosstab to make a 'contingency' table
    observe = pd.crosstab(train.tax_assessed_value_target, train.bathrooms)
    chi2, p, degf, expected = stats.chi2_contingency(observe)
    print('Observed\n')
    print(observe.values)
    print('---\nExpected\n')
    print(expected.astype(int))
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < alpha:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis/")

# 3. Do houses with more `square_feet` have a higher `tax_assessed_value_target`?

def question_3(train, alpha):
    
    # Plot vizual 
    sns.barplot(x = 'square_feet', y='tax_assessed_value_target', data=train, palette='pastel')

    # pandas crosstab to make a 'contingency' table
    observe = pd.crosstab(train.tax_assessed_value_target, train.bathrooms)
    chi2, p, degf, expected = stats.chi2_contingency(observe)
    print('Observed\n')
    print(observe.values)
    print('---\nExpected\n')
    print(expected.astype(int))
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < alpha:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis/")
