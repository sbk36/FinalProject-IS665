#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ### kNN Regression - Diabetes Data 
# ### Lauren Rega
# 
# 

# In[360]:


# First, import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # this is for plotting the data
import seaborn as sns            # this is for visualization

get_ipython().run_line_magic('matplotlib', 'inline')


# In[361]:


#read the file
diabetes = pd.read_csv("diabetes.csv") 


# In[362]:



diabetes.shape # shows shape of dataset


# In[363]:


diabetes.head() #shows first five rows


# In[364]:


diabetes.info() #shows attributes and data type


# In[365]:


# First, use the "isnull()" method to locate the null values if any...
diabetes.isnull()


# In[366]:


# No null values! Let's look at summary statistics

diabetes.describe() 
# Notice that some of these values cannot be zero (Glucose level, BloodPressure, SkinThickness, Insulin, BMI) 
# So we will need to clean the dataset. There are also some maximum values that are far away from the mean. 
# We will take a further look to gain more insight.


# In[367]:


# A simple barchart of our label, 'Outcome', will give us some perspective on the number of females
#who did & did not have diabetes
sns.countplot(x='Outcome',data=diabetes)


# In[368]:


#lets take a closer look at the distribution plots for our independent variables:
sns.displot(diabetes['Pregnancies'])
sns.displot(diabetes['Glucose'])
sns.displot(diabetes['BloodPressure'])
sns.displot(diabetes['SkinThickness'])
sns.displot(diabetes['Insulin'])
sns.displot(diabetes['BMI'])
sns.displot(diabetes['DiabetesPedigreeFunction'])
sns.displot(diabetes['Age'])


# In[369]:


#As suspected some of the attributes have zero's which are skewing the distribution. Let's clean the dataset. 
diabetes['Glucose'] = diabetes['Glucose'].replace(0,diabetes['Glucose'].mean())
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0,diabetes['BloodPressure'].mean())
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0,diabetes['SkinThickness'].mean())
diabetes['Insulin'] = diabetes['Insulin'].replace(0,diabetes['Insulin'].mean())
diabetes['BMI'] = diabetes['BMI'].replace(0,diabetes['BMI'].mean())


# In[370]:


#Now that the zero's are fixed, lets take another look at the distribution plots for our independent variables:
sns.displot(diabetes['Pregnancies'])
sns.displot(diabetes['Glucose'])
sns.displot(diabetes['BloodPressure'])
sns.displot(diabetes['SkinThickness'])
sns.displot(diabetes['Insulin'])
sns.displot(diabetes['BMI'])
sns.displot(diabetes['DiabetesPedigreeFunction'])
sns.displot(diabetes['Age'])


# In[371]:


# some of these features still looked skewed. Let's remove outliers using the Interquartile Range method
# First, define a function called 'outliers' which returns an index of outliers
#IQR = Q3-Q1
# +/- 1.5*IQR
def outliers(df, ft): # inputs are dataframe and feature
    Q1 = df[ft].quantile(0.25) #calculate first quantile
    Q3 = df[ft].quantile(0.75) #calculate third quantile
    IQR = Q3 - Q1 # calculate IQR
    #calc lower & upper bound: 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # create a list which will store the indexes of the outliers
    ls = df.index[ (df[ft] < lower_bound) | (df[ft] > upper_bound)] # if conditions return true, it is outlier
    
    return ls # returns list


# In[372]:


# Now, create an empty list to store output indicies from multiple columns

index_list = []
for feature in['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']:
    index_list.extend(outliers(diabetes, feature))
    


# In[373]:


# Let's see our index list
index_list


# In[374]:


# Next, define a function called 'remove' which returns a cleaned dataframe without outliers

def remove(df, ls):
    ls = sorted(set(ls)) #organize list
    df = df.drop(ls) # drop values
    return df


# In[375]:


df_cleaned = remove(diabetes, index_list)


# In[376]:


df_cleaned.shape # our row count has reduced!


# In[377]:


#Lets save the data to new csv
df_cleaned.to_csv('diabetes_cleaned.csv', index=False)


# In[378]:


#read the file
diabetes_cleaned = pd.read_csv("diabetes_cleaned.csv") 


# In[379]:


diabetes_cleaned.describe() 
#Lets see what our new summary statistics look like


# In[380]:


diabetes_cleaned.shape # Checking the shape again


# In[381]:


#lets see what our distribution plots look like now that we have removed outliers:
sns.displot(diabetes_cleaned['Pregnancies'])
sns.displot(diabetes_cleaned['Glucose'])
sns.displot(diabetes_cleaned['BloodPressure'])
sns.displot(diabetes_cleaned['SkinThickness'])
sns.displot(diabetes_cleaned['Insulin'])
sns.displot(diabetes_cleaned['BMI'])
sns.displot(diabetes['DiabetesPedigreeFunction'])
sns.displot(diabetes_cleaned['Age'])
sns.displot(diabetes_cleaned['Outcome'])


# In[382]:


#Looks great! Now lets start our kNN model.
#First Define X & y:
X = diabetes_cleaned.drop(columns = ['Outcome'])
y = diabetes_cleaned['Outcome']


# In[383]:


from sklearn.neighbors import KNeighborsClassifier


# In[384]:


# Now we scale our data so that the algorithm will perform better
scalar = StandardScaler()
X_scalar = scalar.fit_transform(X)


# In[385]:


#Lets see what our scaled data looks like:
X_scalar


# In[386]:


# Splitting data into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scalar, y, random_state = 100)


# In[387]:


# Creating a classifier object in sklearn
knn = KNeighborsClassifier(p=1)


# In[388]:


# Fitting our model
knn.fit(X_train, y_train)


# In[389]:


# Making predictions
predictions = knn.predict(X_test)
print(predictions)


# In[390]:


# Measuring the accuracy of our model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


# In[391]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[392]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
conf_mat = confusion_matrix(y_test, predictions)
conf_mat


# In[393]:


tpr = conf_mat[0][0]
fpr = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[404]:


# auc scores
auc_score = roc_auc_score(y_test, predictions)

print(auc_score)

# roc curve for models
fpr, tpr, thresh = roc_curve(y_test, predictions, pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.plot(fpr, tpr, linestyle='--',color='darkgreen', label='kNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend()
plt.show()


# In[395]:


# Creating a dictionary of parameters to use in GridSearchCV
from sklearn.model_selection import GridSearchCV

params = {
    'n_neighbors':  range(1, 15, 2),
    'p': [1,2],
    'weights': ['uniform', 'distance']
}


knn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1,
)

knn.fit(X_train, y_train)
print(knn.best_params_)# returns best parameters according to gridsearch

# Returns: {'n_neighbors': 5, 'p': 1, 'weights': 'uniform'}


# In[396]:


# we will use the best parameters in our k-NN algorithm and check if accuracy is increasing.

knn = KNeighborsClassifier(n_neighbors = 11, p =1, weights = 'uniform')


# In[397]:


knn.fit(X_train,y_train)


# In[398]:


knn.score(X_train,y_train)


# In[399]:


knn.score(X_test,predictions)


# In[400]:


predictions = knn.predict(X_test)


# In[348]:


print(classification_report(y_test , predictions))

#you can see that our f1 scores and accuracies have decreased after running cross validation


# In[402]:


# Area Under Curve
auc = roc_auc_score(y_test, predictions)
auc


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, predictions)
plt.plot(fpr, tpr, color='darkgreen', label='ROC')
plt.plot([0, 1], [0, 1], color='orange', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:




