import numpy as np
import pandas as pd
import sklearn.linear_model as LinearRegression

# Dictionary

input_data = {
  "attendence": [90, 85, 95, 80, 70],
  "assignment_score": [85, 90, 80, 75, 95],
  "study_hours": [2, 3, 4, 5, 6],
 }

output_data = {
  "final_grade": [88, 92, 85, 78, 95],
}
 

df = pd.DataFrame(input_data)
df['final_grade'] = output_data['final_grade']

X=df[['attendence', 'assignment_score', 'study_hours']]
y=df[['final_grade']]

print(X)
print(y)

model=LinearRegression.LinearRegression()

model.fit(X,y)

model.predict([[85, 90, 4]])

print("Predicted Final Grade for 85% attendance, 90 assignment score and 4 study hours:", model.predict([[85, 90, 4]])[0])

print("Model Coefficient:", model.coef_[0])  
print("Model Intercept:", model.intercept_[0])
