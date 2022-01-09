# -*- coding: utf-8 -*-
"""
EDA / VDA Project - Diamonds
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# read data from csv and load data in a dataframe
df = pd.read_csv("C:/python_anaconda/python project/diamonds-m (1).csv")

df.head()

# given - column “price” depends on all the other column

''' Q.1 *- What is structure of the dataset? ''' 

# info df
print(df.info())

# summary df
print(df.describe())

''' Q.2 *- What are the data type of each columns? '''

datatype=df.dtypes
print(datatype)

''' Q.3 *- What is the length of alpha-numeric columns? '''
count = 0
for i in df.columns:
    if i.isalnum():
        count = count+1
print(count)        


''' Q.4 *- What are precision & scale of numeric columns? '''

# precision - precision is the number of digits in a number.
# scale - scale is the number of digits to the right of the decimal point in a number.

def precision_and_scale(x):
    a = x
    if isinstance(a, int) :
        if a > 0 :
            precision = len(str(a))
            scale = 0
            return precision, scale
        else:
            precision = 0
            scale = 0

    else:
        if a < 1:
            precision = 0
            scales = str(a).split('.')
        else:
            precision = len(str(a).replace('.', ''))
            scales = str(a).split('.')
        
        try:
            if int(scales[1]) > 0:
                scale = len(scales[1])
            else :
                scale = 0
            
        except IndexError:
            scale = 0
        
        scales.clear()
        return precision, scale



dff1 = df.select_dtypes(include=['int32', 'int64', 'float'])
numeric_column = dff1.columns
print("\tColumns\t\tPrecision\tScale")
prec_l = []
scale_l = []
for i in range(0, len(numeric_column)-1):
    prec_l.clear()
    scale_l.clear()
    for j in dff1[numeric_column[i]]:
        precision, scale = precision_and_scale(j)
        prec_l.append(precision)
        scale_l.append(scale)
    txt = "{:>10}{:>10}{:>10}"
    print(txt.format(numeric_column[i], precision, scale))

''' Q.5 *- For each column,find out
          
          1) Number of Null values
          2) Number of zeros
          3) Provide the obvious errors
          4) Identify columns which should not be alpha-numeric.Provide techniques to fix the same.
'''
''' 1) Number of Null values '''

print(df.isnull().sum()) 

''' 2) Number of zeros '''

print((df==0).sum())

''' 3) Provide the obvious errors '''

obv_error = dict()
for col in df.select_dtypes('object').columns:
    print(col)
    error_list = []
    print(f"** Obvious error in {col} ** \n")
    for val in df[col].unique():
        if ((df[col]==val).sum()) < (0.01* df[col].count()):
            error_list.append(val)
            print(val)
    obv_error[col] = error_list
    if error_list == []:
        print("no obvious errors \n")
print(obv_error)


''' 4) Identify columns which should not be alpha-numeric.Provide techniques to fix the same. '''

# converting z column from object to float64
df['z']=pd.to_numeric(df['z'],errors = 'coerce')

datatype=df.dtypes
print(datatype)

''' Q.6 *- For each numeric column

          1) Replace zero values with suitable statistical value of the column.Give reason why
          2) Replace null values with lower of mean & median value of the column.
          3) Provide the quartile summary along with the count,mean & sum
          4) Provide the range,variance and standard deviation
          5) Provide the count of outliers and their value.Provide a mechanism to fix the outliers
'''

''' 1) Replace zero values with suitable statistical value of the column.Give reason why '''

# replacing zeros with means

# First you can find the nonzero mean :
nonzero_mean = df[ df.x !=0 ].mean()
nonzero_mean1 = df[ df.y !=0 ].mean()
# Then replace the zero values with this mean :
df.loc[ df.x == 0, "x" ] = nonzero_mean 
df.loc[ df.y == 0, "y" ] = nonzero_mean1   

''' 2) Replace null values with lower of mean & median value of the column. '''

# for carat
Ncarat = min(df['carat'].mean(),df['carat'].median())
df['carat'] = df['carat'].fillna(Ncarat)

# for price
Nprice = min(df['price'].mean(),df['price'].median())
df['price'] = df['price'].fillna(Nprice)

'''  3) Provide the quartile summary along with the count,mean & sum '''

# it contains quartile summary along with the count and mean
print(df.describe())

# for sum we refer this
total = df.sum()
print(total)

'''  4) Provide the range,variance and standard deviation '''
# range
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dff = df.select_dtypes(include=numerics)
print(dff.max()-dff.min())
# variance
print(df.var())

# standard deviation
print(df.std())

''' 5) Provide the count of outliers and their value.Provide a mechanism to fix the outliers '''

# count of  outliers
import utils
print(utils.OutlierCount(df))

# now mechanism to fix outliers
df = utils.HandleOutlier(df.select_dtypes(include=numerics))   


''' Q.7 *- For each non-numeric column
          
          1) Replace null values with suitable statistical value of the column.Give reason why
          2) provide frequency distribution table the same
'''

''' 1) Replace null values with suitable statistical value of the column.Give reason why '''

dff = df.select_dtypes('object')
print(dff)
columns = dff.columns
print(columns)
print("Null values in non-numeric column")
print(dff.isnull().sum())
print("After applying statistical value in non numeric column ")
for i in columns:
    df[i].fillna(df[i].mode()[0],inplace=True)
print(df.isnull().sum())

''' 2) provide frequency distribution table the same '''

# for cut
pd.crosstab(df.cut,columns='count')

# for color
pd.crosstab(df.color,columns='count')

# for clarity
pd.crosstab(df.clarity,columns='count')

#for popularity
pd.crosstab(df.popularity,columns='count')

''' Q.8 *- Provide suitable mechanism to convert non numeric columns to numeric. '''

# for 'cut' column
 
print(df['cut'].unique())
from sklearn import preprocessing
leOP = preprocessing.LabelEncoder()
df['cut'] = leOP.fit_transform(df['cut'])
print(df['cut'].unique())

# for color column

print(df['color'].unique())
from sklearn import preprocessing
leOP = preprocessing.LabelEncoder()
df['color'] = leOP.fit_transform(df['color'])
print(df['color'].unique())

# for clarity column

print(df['clarity'].unique())
from sklearn import preprocessing
leOP = preprocessing.LabelEncoder()
df['clarity'] = leOP.fit_transform(df['clarity'])
print(df['clarity'].unique())

# for popularity column

print(df['popularity'].unique())
from sklearn import preprocessing
leOP = preprocessing.LabelEncoder()
df['popularity'] = leOP.fit_transform(df['popularity'])
print(df['popularity'].unique())

''' Q.9 *- Is re-scaling required.If yes,why and what technique would you use for rescaling. '''

 # re-scaling is not required because the data is in same range except id and price but we cannot apply scaling technique on this column because this are depended columns

''' Q.10 *- Provide histogram for all columns.Provide your interpretation on the same. '''

import matplotlib.pyplot as plt

hist = df.columns.tolist()
def make_histo():
    for column in hist:
        if df[column].dtype == 'object':
            continue
        df[column].hist()
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel('frequency')
        plt.show()
        
make_histo()      
  
''' Q.11 *- Provide box & whisker plots for all columns.Provide your interpretation on the same . '''

import seaborn as sns
plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.show()

''' Q.12 *- For numeric columns
         
           1) provide correlation table
           2) provide a suitable graph to visual the same
           3) State which columns be dropped due to multi-collinearity.Give reasons.
'''

''' 1)  provide correlation table '''

pd.options.display.float_format = '{:,.3f}'.format
dfc = df.corr()
print(dfc)  

''' 2) provide a suitable graph to visual the same '''

# plot heat map

print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

''' 3) State which columns be dropped due to multi-collinearity.Give reasons.'''

depVars = 'price'
print("\n*** Multi Colinearity ***")
lMulCorr = utils.MulCorrCols(dfc, depVars, False)   # we exclude longitude and latitude
print(lMulCorr)        
df = df.drop(lMulCorr, axis=1)
print('Done ...')

''' Q.13 *- Prepare relation ship chart showing relation of each numeric column with column “price”.Provide your interpretation on the same. '''


import seaborn
import matplotlib.pyplot as plt
  
df = pd.read_csv("C:/python_anaconda/python project/diamonds-m (1).csv")

depVars = "price"
indepVars = df.columns.difference([depVars])
print(indepVars)
seaborn.pairplot(df,x_vars=indepVars,y_vars=depVars,kind="scatter")
plt.show()

''' Q.14* - Based on feature selection algorithms,identify significant columns of the data set. '''

 # 'id' is a significant column so we can drop column 'id'
print("\n*** Feature Scores - XTC ***")
print(utils.getFeatureScoresXTC(df, depVars))

# pca - select K best 
print("\n*** Feature Scores - SKB ***")
print(utils.getFeatureScoresSKB(df, depVars))

 
''' Q.15 * - Refer to formula of “depth” given above,compute a column 
           “computed depth” based on formula given for each row.Identify 
           or flag the records for which difference between“computed depth”
           & “depth” is greater than 5% Of “depth”.
'''          
df['computed depth'] = df['z']/((df['x']+df['y'])/2)
df['Flag'] = np.where((df['computed depth']-df['depth']) > (5/df['depth'])*100 , 1,0)
df.head()

