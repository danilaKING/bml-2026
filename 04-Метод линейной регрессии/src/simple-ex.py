import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([
    [30],
    [40],
    [50]
])

y = np.array([5, 6, 8])

model = LinearRegression()
model.fit(X, y)

print("Свободный коэффициент:", model.intercept_)
print("Коэффициент площади:", model.coef_[0])

prediction = model.predict([[60]])

print("Цена квартиры 60 м²:", prediction[0])

# визуализация
x_line = np.linspace(25, 65, 100).reshape(-1, 1)
y_line = model.predict(x_line)

x_new = np.array([[60]])
y_new = model.predict(x_new)

plt.scatter(X[:, 0], y, label="Исходные данные")
plt.plot(x_line[:, 0], y_line, label="Линейная регрессия")
plt.scatter(
    x_new[:, 0],
    y_new,
    s=80,
    label="Прогноз для 60 м²"
)

plt.xlabel("Площадь, м²")
plt.ylabel("Цена, млн руб.")
plt.legend()
plt.show()
