import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.DataFrame({
    "distance_km": [
        2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13
    ],

    "items": [
        1, 4, 2, 6, 3, 5,
        2, 7, 4, 6, 3, 8
    ],

    "rush_hour": [
        0, 0, 1, 0, 1, 0,
        1, 0, 1, 1, 0, 1
    ],

    "delivery_time": [
        22, 29, 40, 39, 48, 47,
        55, 57, 65, 70, 60, 79
    ]
})

X = data[[
    "distance_km",
    "items",
    "rush_hour"
]]

y = data["delivery_time"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Свободный коэффициент:")
print(model.intercept_)

print("Коэффициенты признаков:")
for feature, coef in zip(
    X.columns,
    model.coef_
):
    print(feature, coef)

print(
    "MAE:",
    mean_absolute_error(y_test, y_pred)
)

new_order = pd.DataFrame({
    "distance_km": [8],
    "items": [4],
    "rush_hour": [1]
})

prediction = model.predict(new_order)

print("Прогноз времени доставки:", prediction[0])
