#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
# import sklearn
# from sklearn import LinearRegression()


# In[2]:


df1 = pd.read_csv("Chennai houseing sale.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.groupby('AREA')['AREA'].agg('count')


# In[5]:


df2 = df1.drop(['PRT_ID','DATE_SALE','DIST_MAINROAD','PARK_FACIL','QS_ROOMS','QS_BATHROOM','QS_BEDROOM','MZZONE','STREET'],axis='columns')
df2.head()


# In[6]:


df3 = df2.drop(['QS_OVERALL','N_BEDROOM','N_BATHROOM','REG_FEE','COMMIS','UTILITY_AVAIL'],axis='columns')
df3.head()


# In[7]:


df3.shape


# In[8]:


df3.isnull().sum()


# In[9]:


df3['AREA'].unique()


# In[10]:


df3.DATE_BUILD.unique()


# In[11]:


import datetime


# In[12]:


date = pd.DatetimeIndex(df3['DATE_BUILD'])
df3['Year'] = date.year
df3.head()


# In[13]:


df3.head(7)


# In[14]:


df4 = df3.drop(['DATE_BUILD'],axis='columns')
df4.head(11)


# In[15]:


df4.INT_SQFT.unique()


# In[16]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


df4[~df4['INT_SQFT'].apply(is_float)]


# In[18]:


def convert(x):
    return float(x)


# In[19]:


df5 = df4.copy()
df5['INT_SQFT'] = df5['INT_SQFT'].apply(convert)
df5.head()


# In[20]:


df5.loc[107]


# In[21]:


df5.AREA.unique()


# In[22]:


df5['AREA'].replace(['Karapakam','Ana Nagar','Adyr','Velchery','Chrmpet','Chormpet','TNagar','Ann Nagar','KKNagar'],['Karapakkam','Anna Nagar','Adyar','Velachery','Chrompet','Chrompet','T Nagar','Anna Nagar','KK Nagar'],inplace=True)


# In[23]:


df5.AREA.unique()


# In[24]:


df5.head(10)


# In[25]:


df6 = df5.copy()
df6['PRICE_PER_SQFT'] = df5['SALES_PRICE']/df5['INT_SQFT']
df6.head()


# In[26]:


df6[df6.INT_SQFT/df6.N_ROOM<300].head()


# In[27]:


df6.shape


# In[28]:


df7 = df6[~(df6.INT_SQFT/df6.N_ROOM<300)]
df7.shape


# In[29]:


df7.PRICE_PER_SQFT.describe()


# In[30]:


def rem(df):
    df_give = pd.DataFrame()
    for x, subdf in df.groupby('AREA'):
        y = np.mean(subdf.PRICE_PER_SQFT)
        z = np.std(subdf.PRICE_PER_SQFT)
        t = subdf[(subdf.PRICE_PER_SQFT>(y-z)) & (subdf.PRICE_PER_SQFT<=(y+z))]
        df_give = pd.concat([df_give,t],ignore_index=True)
    return df_give
df8 = rem(df7)
df8.shape
df8.head(7)


# In[31]:


df8.shape


# In[32]:


df8.PRICE_PER_SQFT.describe()


# In[39]:


def plot(df,AREA):
    a = df[(df.AREA==AREA) & (df.N_ROOM==2)]
    b = df[(df.AREA==AREA) & (df.N_ROOM==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(a.INT_SQFT,a.PRICE_PER_SQFT,color='red',label='2 ROOM', s=50)
    plt.scatter(b.INT_SQFT,b.PRICE_PER_SQFT,marker='+',color='blue',label='3 ROOM', s=50)
    plt.xlabel('Total square feet Area')
    plt.ylabel('Price per square feet Area')
    plt.title(AREA)
    plt.legend()
    
plot(df8,'Adyar')


# In[40]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,10)
plt.hist(df8.PRICE_PER_SQFT,rwidth=0.7)
plt.xlabel('Price_Per_Sqft')
plt.ylabel('count')


# In[42]:


x = pd.get_dummies(df8.AREA)
x.head()


# In[ ]:


df9 = pd.concat([df8,x.drop],axis='columns')

