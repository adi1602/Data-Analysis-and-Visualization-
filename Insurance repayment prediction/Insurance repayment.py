#!/usr/bin/env python
# coding: utf-8

# # Insurance repayment predication
# 

# ## Import all the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import copy


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the data into the notebook

# In[2]:


data = pd.read_csv(r'D:\insurance.csv') # importing a file


# In[3]:


data.head() 


# In[4]:


data.isnull().sum()  #null value check


# In[5]:


data.shape


# In[6]:


data.describe() # five point summary of the continuous attributes


# In[7]:


#Plots to see the distribution of the continuous features individually

plt.figure(figsize = (15,10))
plt.subplot(231)
plt.hist(data.age,edgecolor = 'blue',color = 'green',alpha=0.7)
plt.xlabel('Age')

plt.subplot(232)
plt.hist(data.bmi,color = 'lightblue',edgecolor = 'black',alpha=0.9)
plt.xlabel('BMI')

plt.subplot(233)
plt.hist(data.charges, color = 'brown',edgecolor = 'black',alpha=0.8)
plt.xlabel('Charges')
plt.show()


# In[9]:


Skewness = pd.DataFrame({'Skewness' : [stats.skew(data.bmi),stats.skew(data.age),stats.skew(data.charges)]},
                        index=['bmi','age','charges'])
print(Skewness)


# ## Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

# In[10]:


plt.figure(figsize=(15,10))
plt.subplot(221)
sns.boxplot(x = data.age,color='grey')

plt.subplot(222)
sns.boxplot(x = data.bmi,color='grey')

plt.subplot(223)
sns.boxplot(x = data.charges,color='grey')
plt.show()


# In[11]:


data.smoker.value_counts()


# In[12]:


plt.figure(figsize=(20,25))

x = data.smoker.value_counts().index
y = [data['smoker'].value_counts()[i] for i in x]
plt.subplot(421)
plt.bar(x,y, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)
plt.xlabel('Smoker')
plt.ylabel('Count ')
plt.title('Smoker distribution')

x1 = data.sex.value_counts().index
y1 = [data['sex'].value_counts()[i] for i in x1]
plt.subplot(422)
plt.bar(x1,y1, align='center',color='lightblue',edgecolor='black')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Sex Distribution')

x2 = data.children.value_counts().index
y2 = [data['children'].value_counts()[i] for i in x2]
plt.subplot(423)
plt.bar(x2,y2, align='center',color = 'lightblue',edgecolor = 'black')
plt.xlabel('Number of children')
plt.ylabel('Count')
plt.title('Childrens')

x3 = data.region.value_counts().index
y3 = [data['region'].value_counts()[i] for i in x3]
plt.subplot(424)
plt.bar(x3,y3, align='center',color = 'lightblue',edgecolor = 'black')
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('regions')
        
plt.show()


# In[18]:


data_encoded = copy.deepcopy(data)
data_encoded.loc[:,['sex', 'smoker','region']] = data_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 

sns.pairplot(data_encoded)  
plt.show()


# In[19]:


data_encoded


# In[27]:


plt.figure(figsize=(10,5))
sns.scatterplot(data.age,data.charges,hue=data.smoker,palette= ['red','green'] ,alpha=0.6)
plt.show()


# In[28]:


plt.figure(figsize=(10,5))
sns.scatterplot(data.age,data.charges,hue=data.sex,palette=['lightpink','grey'])
plt.show()


# In[ ]:




