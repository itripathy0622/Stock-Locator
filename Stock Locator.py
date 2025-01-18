#!/usr/bin/env python
# coding: utf-8

# Exporatory Data Analysis

# ### Packages

# In[6]:


get_ipython().system('pip install yfinance')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime, timedelta
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FuncFormatter


# ## Preliminary Visualization

# In[18]:


symbolList =  ['IBM', 'MAT', 'NFLX']
START_DATE = '2023-07-26' 
END_DATE = '2023-08-02' 


# In[6]:


stockPxList = yf.download(symbolList, START_DATE, END_DATE)['Adj Close']
stockLogRetList = np.log(stockPxList).diff().dropna()


# Visualization #1: 

# In[7]:


num=0
while num<len(symbolList):
    plt.subplots()
    Points=stockPxList[symbolList[num]]
    Points.plot(title = symbolList[num] + ' Daily Prices', ylabel = 'Price in USD ($)', color = 'b')
    num=num+1


# Visualization #2:

# In[8]:


num=0
while num<len(symbolList):
    plt.subplots()
    logRetPoints=stockLogRetList[symbolList[num]]
    logRetPoints.plot(title = symbolList[num] + ' Daily Log Returns', ylabel = 'Log Return', color = 'b')
    num=num+1



# ## Preliminary Normality Testing

# In[9]:


num=0
while num<len(symbolList):
    shapiroTest = stats.shapiro(stockLogRetList[symbolList[num]])
    print(symbolList[num] + ' :',shapiroTest.pvalue)
    num+=1



# In[11]:


from scipy.stats import norm, t
mu = 0  
sigma = 1 
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Normal pdf')

df = 2.74
mean, var, skew, kurt = t.stats(df, moments='mvsk')
x = np.linspace(t.ppf(0.01, df),t.ppf(0.99, df), 100)
plt.plot(x, t.pdf(x, df), label='Student-t pdf')
plt.title("Normal vs Student-t Distributions")
plt.xlabel("Values"); plt.ylabel("Density")
plt.legend()


# In[14]:


jarque_bera_test = stats.jarque_bera(stockLogRetList)
jarque_bera_test.pvalue


# ## Preliminary Pre-processing

# In[13]:


TICKER = 'MAT' 


# In[14]:


FEATURES = symbolList.copy()
stockPx = yf.download(TICKER, START_DATE, END_DATE)['Adj Close'] 
stockPx01 = (stockPx.pct_change().dropna() > 0).astype(int)
ax = sns.countplot(x = stockPx01)
plt.title('Directional (Up=1/Down=0) Distribution')
plt.xlabel(TICKER + ' Direction')
plt.ylabel('Count')
total = len(stockPx01)
for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x_coord = p.get_x() 
        y_coord = p.get_y() + p.get_height()+0.02
        ax.annotate(percentage, (x_coord, y_coord))


# In[15]:


import pandas_datareader as pdr

INDICATORS = ['JHDUSRGDPBR', 'T10Y3M', 'NFCI', 'NFCINONFINLEVERAGE', 'UMCSENT', 'GDPC1']
VAR_NAMES = ['recession', 'yield_curve', 'financial_conditions', 'leverage', 'sentiment', 'real_gdp']
FEATURES = VAR_NAMES[1:]
# what we are predicting - the recession status
RESPONSE = VAR_NAMES[0]
VAR_DISPLAY = ['Recession', 'Yield Curve', 'Financial Conditions', 'Leverage', 'Sentiment', 'Real GDP']
col_dict = dict(zip(VAR_NAMES, VAR_DISPLAY))

# putting them all into one datafram'e
econ = (pdr.DataReader(INDICATORS, 'fred', 1980, 2023)
        .ffill()
        .resample('M')
        .last()
        .dropna())
econ.columns = VAR_NAMES

econ.head(10)


# In[16]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.DataFrame({'Features': FEATURES, 'VIF': [variance_inflation_factor(econ.loc[:, FEATURES].values, i) for i in range(len(FEATURES))]})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(stockLogRetList, stockPx01, test_size=0.2, random_state=0)


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(econ.loc[:, FEATURES], econ.loc[:, RESPONSE], test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier 

rf_model = RandomForestClassifier(random_state=0) 

rf_model.fit(X_train, y_train) 

importances = rf_model.feature_importances_ 

indices = np.argsort(importances) # sort the features' index by their importance scores

plt.title('Feature Importances in the Random Forest Model')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
plt.xlabel('Importance Score')


# The features are ranked from most important to least important as financial conditions (the current state of the market), sentiment (consumer enjoyment), leverage (tightness or looseness of the market), real_gdp (money in the economy), and yield curve (interest rate of bonds). 
