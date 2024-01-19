# Importing libraries
import pandas as pd
import numpy as np

# Loading the dataset
df = pd.read_csv("Weather.csv")
print(df)

# To print first five rows of dataset
print(df.head())

# To print shape of dataframe
print(df.shape)

# To show the index of dataframe
print(df.index)

# To display the name of each column
print(df.columns)

#To show the datatype of each column
print(df.dtypes)

# To show the unique values
print(df['WindGustDir'].unique())

# To show the total no. of unique values in each column
print(df.nunique())

# To show the total number of non-null values in each column
print(df.count())

# To show all unique values with their counts
print(df['WindGustDir'].value_counts())

# To provide basic information about dataframe
print(df.info())

# To find all the unique "WindSpeed3pm" values in the data
print(df.head(2))
print(df['WindSpeed3pm'].nunique())
print(df['WindSpeed3pm'].unique())  #answer

# Find the number of time when the "WindGustDir is in E (East)"
print(df.head(2))

#value_counts()
print(df.WindGustDir.value_counts())

#Filtering
print(df[df.WindGustDir == 'E'])

#groupby()
print(df.groupby('WindGustDir').get_group('E'))

# Find the number of times when the 'WindSpeed9am was exactly 4 km/hr"
print(df.head(2))

print(df[df['WindSpeed9am'] == 4]) #Answer

# Find out the null values in the data
print(df.isnull().sum())

# what is the mean "WindGustSpeed"
print(df.WindGustSpeed.mean())

# What is the standard deviation of "Pressure3pm" in this data
print(df.Pressure3pm.std())

# what is the variance of "Humidity9am" in this data
print(df['Humidity9am'].var())

# Find all instances when "WNW" was recorded in WindDir9am
#filtering
print(df[df['WindDir9am'] == 'WNW'])

# Find all instances when "WindSpeed3pm is above 24" and "Cloud9am is 8"
print(df.head(2))

print(df[(df['WindSpeed3pm'] > 24) & (df['Cloud9am'] == 8)])

# show all records where RainTomorrow is Yes
print(df[df['RainTomorrow'] == 'Yes'])

#Find all instances when 'RainToday is No' or 'RISK_MM is above 22.6'
print(df[(df['RainToday'] == 'No') | (df['RISK_MM'] > 22.6)])

