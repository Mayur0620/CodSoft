#!/usr/bin/env python
# coding: utf-8

# # IRIS Flower Classification

# **Project by: Mayur Rajput**

# ### Project Goal:

# The Goal of this project is to build a machine learning model that can accurately classify species of the Iris flowers.

# ### Project Description :

# The "IRIS Flower Classification" project aims to utilize machine learning techniques and Python programming to build a model capable of accurately classifying Iris flowers into their respective species. The Iris flower dataset, which comprises three species (setosa, versicolor, and virginica), will serve as the foundation for this classification task. By analyzing the sepal and petal measurements of these flowers, the project will develop a robust classification model.

# #### Here’s a step-by-step breakdown of what we’ll do :

# **Data Collection:** Obtain the Iris dataset from kaggle. Ensure the dataset includes measurements of sepal and petal features for three Iris species.
# 
# **Data Preprocessing:** Check for missing values in the dataset and decide on the appropriate handling method (imputation, removal, etc.).If any attributes are categorical (though unlikely in the Iris dataset), encode them into numerical values.
# 
# **Feature Engineering:** Given the simplicity of the Iris dataset, feature engineering may not be extensive. Consider standardizing or scaling the features to ensure uniform influence in the model.
# 
# **Data Visualization:** Visualize the data, including scatter plots or histograms, to gain insights into how features relate to species classification. Explore any patterns or correlations in the data.
# 
# **Model Selection:** Select a classification algorithm suitable for the task, such as logistic regression, decision trees, or support vector machines.
# 
# **Model Training:** Split the dataset into training and testing subsets for model evaluation. Train the chosen model using the training data.
# 
# **Model Evaluation:** Evaluate the model's performance using classification metrics like accuracy, precision, recall, and F1-score. Choose the model that offers the most accurate Iris flower classification.
# 

# ## 1 - Importing Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## 2 - Importing Dataset

# In[2]:


data = pd.read_csv('IRIS.csv')


# In[3]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\nborder: 1px solid black !important;\n}\n</style>')


# ## 3 - Data Exploration

# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.info()


# **3.1 - Identifying Missing Values in Dataset**

# In[8]:


data.isnull().mean()*100


# **3.2 - Removing Duplicate Values from Dataset**

# In[9]:


data.drop_duplicates(inplace=True)


# In[10]:


data.shape


# In[11]:


#Removing substring "Iris-" from Species Column
data['species'] = data['species'].str.replace('Iris-','', regex = True)


# In[12]:


data.head()


# ## 4 - Data Visualization

# **Sepal Length & Petal Length by Species**

# In[13]:


sns.set_theme()


# In[14]:


sns.scatterplot(x='sepal_length', y ='petal_length', hue='species', data=data)
plt.title('Sepal Length & Petal Length by Species')
plt.show()


# **Sepal Width & Petal Width by Species**

# In[15]:


sns.scatterplot(x= 'sepal_width', y= 'petal_width', hue = 'species', data=data)
plt.title('Sepal Width & Petal Width by Species')
plt.show()


# **Species Distribution**

# In[16]:


sns.countplot(x='species',data=data)


# In[17]:


print(data['species'].value_counts())


# **Correlation matrix**

# In[18]:


correlation = data.corr()


# In[19]:


plt.figure(figsize=(5,4))
sns.heatmap(correlation, annot=True, cmap='Blues', fmt=".2f")
plt.show()


# ## 5 - Data Preprocessing

# **5.1 - Label Encoding**

# In[20]:


label = LabelEncoder()


# In[21]:


data['species'] = label.fit_transform(data['species'])


# In[22]:


data.head()


# In[23]:


data.tail()


# 0 -> Setosa 
# 
# 1 -> Versicolor
# 
# 2 -> Virginica

# **5.2 - Separating feature variables and target variable**

# In[24]:


x = data.drop(columns=['species'], axis = 1)
y = data['species']
print(x)


# In[25]:


print(y)


# **5.4 - Spliting data into Training & Testing data**

# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2)


# In[27]:


print(x.shape, x_train.shape, x_test.shape)


# ## 6 - Model Building & Training

# In[28]:


model = LogisticRegression()


# In[29]:


model.fit(x_train, y_train)


# ## 7 - Model Evaluation

# In[30]:


# Accuracy of training data
x_train_prediction = model.predict(x_train)

accuracy_train = accuracy_score(x_train_prediction, y_train)
print("Accuracy Score of training data:",accuracy_train)


# In[31]:


# Accuracy of test data
x_test_prediction = model.predict(x_test)

accuracy_test = accuracy_score(x_test_prediction, y_test)
print("Accuracy Score of test data:",accuracy_test)


# ## 8 - Model Deployment

# In[32]:


# Create an empty list to store user input
input_data = []

# Define the list of features (replace with your actual feature names)
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Loop through each feature to get user input
for feature in features:
    while True:
        try:
            value = float(input(f"Enter {feature}: "))
            input_data.append(value)
            break  # Exit the loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Convert the list to a NumPy array
input_data_as_np_array = np.array(input_data)

# Now, user_input_as_np_array contains the input data as a NumPy array
print("User input:")
print(input_data_as_np_array)


# In[34]:


# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_np_array.reshape(1, -1)

# Make a prediction using the model
prediction = model.predict(input_data_reshaped)
print(prediction)

# Map the predicted label to the actual species name
if (prediction[0] == 0):
    print('Setosa')
elif (prediction[0] == 1):
    print('Versicolor')
else:
    print('virginica')


# In[ ]:




