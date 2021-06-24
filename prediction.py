
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:20:55 2020

@author: clay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn import svm, feature_selection, linear_model 
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import r2_score

# Reading the file 
dataset = pd.read_csv('Manhattan12.csv', header = 4)

def summary():
    global dataset
    # Will print list of missing values
    pd.isnull(dataset).sum()
    # Will provide summary for the sale price column
    dataset['SALE PRICE'].describe().apply(lambda x: '%.5f' % x)

def inspect():
    global dataset
    
    # Inspecting the datatypes of all columns
    dataset.dtypes
    
    # Changing specific object columns to numerical values
    for col in ["RESIDENTIAL UNITS", "TOTAL UNITS", "LAND SQUARE FEET", "GROSS SQUARE FEET"]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
        
    # Renaming specific volumns
    dataset.rename(columns={'SALE\nPRICE': 'SALE PRICE','APART\nMENT\nNUMBER': 'APARTMENT NUMBER'}, inplace=True)
        
    dataset["SALE DATE"] = pd.to_datetime(dataset["SALE DATE"], errors="coerce")
    dataset["SALE PRICE"] = dataset["SALE PRICE"].replace('[\$,]', '', regex=True).astype(float)
    dataset['BUILDING AGE'] = dataset['SALE DATE'].dt.year - dataset['YEAR BUILT']
    
    # Merging serveral column values together 
    dataset.loc[dataset['NEIGHBORHOOD'].str.contains('UPPER EAST'), 'NEIGHBORHOOD'] = 'UPPER EAST SIDE'
    dataset.loc[dataset['NEIGHBORHOOD'].str.contains('UPPER WEST'), 'NEIGHBORHOOD'] = 'UPPER WEST SIDE'
    dataset.loc[dataset['NEIGHBORHOOD'].str.contains('MIDTOWN'), 'NEIGHBORHOOD'] = 'MIDTOWN'
    dataset.loc[dataset['NEIGHBORHOOD'].str.contains('GREENWICH'), 'NEIGHBORHOOD'] = 'GREENWICH VILLAGE'
    dataset.loc[dataset['NEIGHBORHOOD'].str.contains('HARLEM-CENTRAL|HARLEM-UPPER|HARLEM-WEST'), 'NEIGHBORHOOD'] = 'HARLEM'
    
    # Converting object columns to category
    for col in ['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']:
        dataset[col] = dataset[col].astype('category')
    
    # Dropping several values which will be useless for data analysis
    dataset = dataset.drop(["EASE-MENT","BOROUGH","GROSS SQUARE FEET","LAND SQUARE FEET"], axis=1)
    
def visualise():
    global dataset
    
    # Finding correlating values in dataset
    plt.figure(figsize=(20,10))
    c = dataset.corr()
    
    # Getting ordered list of features that correlate with sale price
   # print(c['SALE PRICE'].sort_values(ascending=False))
    
    # Getting heatmap of feature correlation
    #sns.heatmap(c,cmap="BrBG",annot=True)
    
    # Bar chart of tax class at present vs sales price
    #pivot = dataset.pivot_table(index='TAX CLASS AT PRESENT', values='SALE PRICE', aggfunc=np.median)
    #pivot.plot(kind='bar', color='black')
    
    #sns.pairplot(dataset)
    #plot_cols = ["RESIDENTIAL UNITS","COMMERCIAL UNITS", "TOTAL UNITS", "YEAR BUILT", "SALE\nPRICE"]
    #fig = plt.figure(1, figsize=(12, 12))
    #fig.clf()
    #ax = fig.gca()
    #scatter_matrix(dataset[plot_cols], alpha=0.3, diagonal='hist', ax = ax) 
    #plt.show()
    
    # Creating scatter plot of building age vs sales price
    #sns.scatterplot(x='BUILDING AGE', y='SALE PRICE', data=dataset)

    # Bar chart of how sales prices is different across neighborhoods
    pivot = dataset.pivot_table(index='NEIGHBORHOOD', values='SALE PRICE', aggfunc=np.median)
    pivot.plot(kind='bar', color='Blue')

    # Visualise if lnprice is skewed
    #sns.distplot(dataset['lnprice'])

def clean():
    global dataset
    
    # Filtering the data based on duplicated
    dataset = dataset.drop_duplicates(subset=None, keep='first', inplace=False)
  
    # Replacing every 0 in cols with NaN
    cols = ["SALE PRICE", "TOTAL UNITS"]
    for col in cols:
        dataset[col] = dataset[col].replace(0, np.nan)
        dataset[col] = dataset[col].dropna()
            
    #print(pd.isnull(dataset).sum())
    
    # Dropping all occurences where null
    dataset = dataset.dropna()
    # Dropping columns which will not help linear model
    dataset = dataset.drop(["ADDRESS", "ZIP CODE", "APARTMENT NUMBER","BLOCK","LOT","SALE DATE","BUILDING CLASS AT PRESENT","BUILDING CLASS AT TIME OF SALE","YEAR BUILT"], axis=1)
    # Removing all columns where total of commercial and residential units is not equal to total units
    dataset = dataset[dataset['TOTAL UNITS'] == dataset['COMMERCIAL UNITS'] + dataset['RESIDENTIAL UNITS']]
    
    dataset['lnprice'] = np.log(dataset['SALE PRICE'])
    # Looking at number of unique values for total units
    dataset['TOTAL UNITS'].value_counts()
    # Dropping total units where value is 1 or less
    dataset.drop(dataset.index[dataset['TOTAL UNITS'] <= 1], inplace = True)    
        
def outliers():
    global dataset
    
    # Loops through and remove outliers for specific columns
    outliers = ['SALE PRICE','lnprice','TOTAL UNITS']
    for col in outliers:
        temp = np.zeros(dataset.shape[0])
        for i, x in enumerate(dataset[col]):
            q1, q3 = np.percentile(dataset[col],[25,75])
            iqr = q3 - q1
            # Makes temp = 1 if outlier detected
            lower_bound = q1 -(1.5 * iqr) 
            upper_bound = q3 +(1.5 * iqr) 
            if (x > upper_bound): temp[i] = 1
            if (x < lower_bound): temp[i] = 1
        # Filter data to remove outliers
        dataset['outlier'] = temp
        dataset = dataset[dataset.outlier == 0]
        dataset.drop('outlier', axis = 1, inplace = True)
        dataset.reset_index(inplace = True, drop = True)
    
    
def encode():
    global dataset
    # Picking categorical columns to encode
    one_hot_features = ['NEIGHBORHOOD','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY']
    # Encoding the categorical values selected 
    one_hot = pd.get_dummies(dataset[one_hot_features],drop_first=True)
    # Dropping original columns from the dataset
    dataset = dataset.drop(one_hot_features,axis = 1)
    # Join the encoded dataset
    dataset = dataset.join(one_hot)
    
def normalise():
    global dataset
    
    # Selecting all the numeric columns
    num_cols = dataset.select_dtypes(include=[np.number]).copy() 
    # Dropping the price column
    num_cols.drop('SALE PRICE', axis='columns', inplace=True) 
    # Normalising the numeric columns
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    # Dropping any NaN columns that have been created from normalising
    df_norm = df_norm.dropna(axis=1, how='any')    
    
    return df_norm

def regression(df_norm):
    global dataset

    df = df_norm.select_dtypes(include=[np.number]).copy() 
    feature_cols = df.columns.values.tolist() 
    feature_cols.remove('lnprice')
    XO = df[feature_cols]
    YO = dataset['lnprice']
    estimator = svm.SVR(kernel="linear")
    
    # Getting 30 most correlating columns
    selector = feature_selection.RFE(estimator, 30, step=1) 
    selector = selector.fit(XO, YO)
    
    # Selecting the most correlating features
    select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist() 
    
    X = df[select_features]
    Y = dataset['lnprice']
    # Splitting dataset into training and testing
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2) 
    lm = linear_model.LinearRegression()
    lm.fit(trainX, trainY)
    # Printing out the R^2 score
    print("R squared for the training data is {:4.3f}".format(lm.score(trainX, trainY))) 
    print("Score against test data: {:4.3f}".format(lm.score(testX, testY)))
    
   
    # Visualising accuracy of R^2 score by plotting predicted vs actual prices
    pred_trainY = lm.predict(trainX)
    plt.figure(figsize=(14, 8))
    plt.plot(trainY, pred_trainY, 'o') 
    plt.ylabel="Predicted Prices"
    plt.title="Plot of predicted vs actual prices" 
    plt.show()

inspect()
clean()
outliers()
visualise()
encode()
df_norm = normalise()
regression(df_norm)











