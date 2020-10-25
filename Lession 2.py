#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
from pandas_datareader import data
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score 
mydata= pd.read_csv('C:/Users/darkp/OneDrive/ly_hoctap/WA_Fn-UseC_-HR-Employee-Attrition.csv')
mydata1 = pd.DataFrame(mydata)


# In[69]:


def quick_analysis(df):
    print("Data type:")
    print(df.dtypes)
    print("row and columns:")
    print(df.shape)
    print("columns name:")
    print(df.columns)
    print("null value:")
    print(df.info())
    print(df.apply(lambda x : sum(x.isnull()) / len(df)))
quick_analysis(mydata1)
#Dataset has 1470 sample and 35 feature with (26 Numerical, 9 object feature). In dataset don't missing value. 


# In[70]:


mydata1.head()
# overview dataset


# In[42]:


categorical_attributes = mydata1.select_dtypes(include =['object'])
numvar = mydata1.select_dtypes(include = 'number')
for i in numvar.columns:
    if len(mydata1[i].unique()) <=10 :
        mydata1[i]=mydata1[i].astype(np.object)
        print(i,len(mydata1[i].unique()),mydata1[i].dtypes)
    else:
        print(i,len(mydata1[i].unique()),mydata1[i].dtypes)
mydata1.info()
#Variable has class <=10 then replace type from number to object 
# After replace has 13 Numerical, 19 object feature


# In[71]:


for i in mydata1.columns:
   if len(mydata1[i].unique()) ==1 :
       print([i])
# delete variables with only class. Remove 3 feature 


# In[72]:


for i in mydata1.columns:
    if len(mydata1[i].unique()) ==1 :
        del mydata1[i]
mydata1


# In[73]:


duplicate_rows_df = mydata1[mydata1.duplicated()]
duplicate_rows_df.shape
# Test dataset has duplicate, result data hasn't duplicate


# In[74]:


mydata1.describe(percentiles =[0.05,0.25,0.5,0.75,0.95])
# feature YearsAtCompany has large range between value at 95%(20) and value max (40)
# feature TotalWorkingYears the same YearsAtCompany
#Therefore, dataset has ability contains outlier.


# In[75]:


import plotly.express as px
for i in numvar:
    fig = px.box (mydata1, x= i) 
    fig.show()
#visualisation plot box with purpose detection oulier


# In[76]:


Q1 = mydata1.quantile(0.25)
Q3 = mydata1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
# Definition values outsise range from Q1 - 1.5 * IQR to mydata1 > (Q3 + 1.5 * IQR ) are considered outlier


# In[77]:


mydata1 = mydata1[~((mydata1 < (Q1 - 1.5 * IQR)) |(mydata1 > (Q3 + 1.5 * IQR))).any(axis=1)]
mydata1.shape
# remore outliner


# In[78]:


df_scaled = mydata1.copy()
features = df_scaled[numvar.columns]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled[numvar.columns] = scaler.fit_transform(features.values)
df_scaled
# Data tranformation by method Standard Scaler


# In[79]:


df_scaled["Attrition"].value_counts()


# In[80]:


#Attrition rate
attritionRate = df_scaled["Attrition"].value_counts() / df_scaled['Attrition'].shape[0]
attritionRate = attritionRate * 100
print("%.3f" % attritionRate[0] + "% is No")
print("%.3f" % attritionRate[1] + "% is Yes" )


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns
def _plot_hist_subplot(x, fieldname, bins = 10, use_kde = True):
  x = x.dropna()
  xlabel = '{} bins tickers'.format(fieldname)
  ylabel = 'Count obs in {} each bin'.format(fieldname)
  title = 'histogram plot of {} with {} bins'.format(fieldname, bins)
  ax = sns.distplot(x, bins = bins, kde = use_kde)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  return ax

# Biểu đồ barchart
def _plot_barchart_subplot(x, fieldname):
  xlabel = 'Group of {}'.format(fieldname)
  ylabel = 'Count obs in {} each bin'.format(fieldname)
  title = 'Barchart plot of {}'.format(fieldname)
  x = x.fillna('Missing')
  df_summary = x.value_counts(dropna = False)
  y_values = df_summary.values
  x_index = df_summary.index
  ax = sns.barplot(x = x_index, y = y_values, order = x_index)
  # Tạo vòng for lấy tọa độ đỉnh trên cùng của biểu đồ và thêm label thông qua annotate.
  labels = list(set(x))
  for label, p in zip(y_values, ax.patches):
    ax.annotate(label, (p.get_x()+0.25, p.get_height()+0.15))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  return ax
fig = plt.figure(figsize=(18, 16))
fig.subplots_adjust(hspace=0.5, wspace=0.2)
# Tạo vòng for check định dạng của biến và visualize
for i, (fieldname, dtype) in enumerate(zip(df_scaled.columns, df_scaled.dtypes.values)):
  if i <= 11:
    ax_i = fig.add_subplot(4, 3, i+1)
    if dtype in ['float64', 'int64']:
      ax_i = _plot_hist_subplot(df_scaled[fieldname], fieldname=fieldname)
    else:
      ax_i = _plot_barchart_subplot(df_scaled[fieldname], fieldname=fieldname)
      
fig.suptitle('Visualization all fields')
plt.show()


# In[55]:


df_scaled['log_DistanceFromHome'] = np.log(df_scaled['DistanceFromHome'])


# In[81]:


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
categorical_attributes_df = df_scaled.select_dtypes(include =['object'])
categorical_attributes_df.columns
numvar_df = [c for c in df_scaled.columns if c not in categorical_attributes ]


# In[82]:


df_scaled1 =df_scaled.copy()
for i in categorical_attributes_df.columns:
    df_scaled1[i] = lb_make.fit_transform(df_scaled1[i])
df_scaled1
    


# In[83]:


df_scaled1.shape


# In[84]:


df_scaled1.info()


# In[85]:


corr = df_scaled1.corr()
corr


# In[87]:


fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="Blues",linewidth=0.3)


# In[89]:


categorical_number_2 = df_scaled.select_dtypes(exclude = 'number').drop('Attrition', axis =1 ).columns
categorical_number_2


# In[90]:


from scipy import stats
from scipy.stats import chi2_contingency
for i in categorical_number_2: 
    chi_res = chi2_contingency(pd.crosstab(df_scaled[i],df_scaled['Attrition']))
    print(i,'Chi2 Statistic: {},p_value:{}'.format(chi_res[0],chi_res[1]))


# In[91]:


chi2_check = []
for i in categorical_number_2: 
    if chi2_contingency(pd.crosstab(df_scaled[i],df_scaled['Attrition']))[1] < 0.05:
        chi2_check.append("Reject null hypothesis")
    else:
        chi2_check.append("Fail to Reject null hypothesis")
res = pd.DataFrame(data = [categorical_number_2,chi2_check]).T
res.columns = ['columns','hypothesis']
print(res)


# In[94]:


df_scaled.info()


# In[95]:


mydata1.info()


# In[104]:


dg = pd.DataFrame(corr.iloc[1])
dg.sort_values(by=['Attrition'])
Choose feature has absolute correlation large (OverTime,StockOptionLevel,TotalWorkingYears,MaritalStatus,YearsAtCompany)


# In[ ]:




