# import modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("Iris.csv")
print(df.head())

# To delete a column
df = df.drop(columns = ['Id'])
print(df.head())

# To display stats about data
print(df.describe())

# To display information about dataset
print(df.info())

# To display no. of samples on each class
print(df['Species'].value_counts())

# Preprocessing the dataset
# check for null values
print(df.isnull().sum())

# Exploratory Data Analysis
#histograms

# df['SepalLengthCm'].hist()
# plt.show()
#
# df['SepalWidthCm'].hist()
# plt.show()
#
# df['PetalLengthCm'].hist()
# plt.show()
#
# df['SepalWidthCm'].hist()
# plt.show()
#
# #scatterplots
# colors = ['purple', 'red', 'green']
# species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
#
# for i in range(3):
#     x = df[df['Species'] == species[i]]
#     plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
#     plt.xlabel("Sepal Length")
#     plt.ylabel("Sepal Width")
#     plt.legend()
#     plt.show()
#
# for i in range(3):
#         x = df[df['Species'] == species[i]]
#         plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])
#         plt.xlabel("Petal Length")
#         plt.ylabel("Petal Width")
#         plt.legend()
#         plt.show()
#
# for i in range(3):
#     x = df[df['Species'] == species[i]]
#     plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
#     plt.xlabel("Sepal Length")
#     plt.ylabel("Petal Length")
#     plt.legend()
#     plt.show()
#
# for i in range(3):
#     x = df[df['Species'] == species[i]]
#     plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
#     plt.xlabel("Sepal Width")
#     plt.ylabel("Petal Width")
#     plt.legend()
#     plt.show()

# Label Encoder
# In ML we deal with datasets which contains multiple labels in one or more than columns. These Labels can be in the form of words or numbers. Label Encoding refers to converting the label into numeric form so as to convert it into machine readable form.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])
print(df.head())

#Model Training
# from sklearn.model_selection import train_test_split
# # train - 70
# # test - 30
# x = df.drop(columns=['Species'])
# y = df['Species']
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0,30)

#logistic regression
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(x_train, y_train)






