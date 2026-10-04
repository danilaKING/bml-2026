import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X = np.array([
    [10], [15], [20], [25],
    [30], [35], [40], [45],
    [50], [55], [60], [65]
])

y = np.array([
    94, 108, 119, 137,
    146, 164, 171, 189,
    202, 214, 231, 239
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

new_budget = np.array([[52]])

print(
    "Прогноз продаж:",
    model.predict(new_budget)[0]
)
