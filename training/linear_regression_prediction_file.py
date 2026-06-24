import pandas as pd
import numpy as np
import sklearn.linear_model as LinearRegression


df=pd.read_csv("training\students_marks.csv")
print(df)

# Data Science Enginer --> I will ceck whther the data is clean(no missing value/no duplicate value/no outliers)

print(df.isnull().sum()) # check for missing values

# I will fill the missing value using calculate Mean--> Average of column

df['hours'].fillna(df['hours'].mean(), inplace=True)
print(df)

# As a Data Science Engineer, I will check for duplicate values in the dataset
print(df.duplicated().sum()) # check for duplicate values

print(df.drop_duplicates()) # drop duplicate values

# m=? ,c=?

# LinearRegression

model=LinearRegression.LinearRegression()
X=df[['hours']]
y=df[['marks']]

#  train the model
model.fit(X,y)

# y=mx+c, m=?, c?

predicted_marks=model.predict([[7]])
print("Predicted Marks for 7 hours of study:", predicted_marks[0])

print("Model Coefficient:", model.coef_[0])
print("Model Intercept:", model.intercept_[0])

