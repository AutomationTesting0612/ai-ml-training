import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data= {
"experince": [1,2,3,4,5],
"Salary": [20000,30000, 45000, 55000, 65000]
}

df = pd.DataFrame(data)
print(df)

X=df[["experince"]]
Y=df[["Salary"]]

poly = PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(X)

model = LinearRegression()

model.fit(poly_features, Y)
new_x = poly.transform([[6]])

prediction =model.predict(new_x)


print("Predicted Salary =", prediction[0])



