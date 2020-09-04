#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


df=pd.read_csv("F:\Downloads_new/news.csv")
df.shape
df.head()


# In[7]:


labels=df.label
labels.head()


# In[14]:


df.dtypes


# In[13]:


x_train, x_test, y_train, y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=10)
print(x_train.shape)
print(x_test)
#print(x_train)


# In[34]:


x_train.dtypes


# In[64]:


df1=pd.read_csv("C:/Users/Aprajita/Downloads/Test1.csv")
df1.shape
df1.head()


# In[65]:


df1.dtypes


# In[66]:


my_set=df1['text']
my_set.dtypes


# In[67]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[ ]:





# In[68]:


passive_aggressive=PassiveAggressiveClassifier(max_iter=50)
passive_aggressive.fit(tfidf_train,y_train)
y_pred=passive_aggressive.predict(tfidf_test)
print(accuracy_score(y_test,y_pred))


# In[69]:


print(y_pred)


# In[70]:


my_set_tfidf_test=tfidf_vectorizer.transform(my_set)


# In[71]:


my_set_pred=passive_aggressive.predict(my_set_tfidf_test)
print(my_set_pred)


# In[18]:


confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])


# In[19]:


print(y_test)


# In[ ]:





# In[ ]:





# In[ ]:




