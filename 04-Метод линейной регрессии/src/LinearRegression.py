import numpy as np


class MyLinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        ones = np.ones((X.shape[0], 1))
        X_matrix = np.hstack((ones, X))

        X_T = X_matrix.T

        X_TX = X_T @ X_matrix

        X_Ty = X_T @ y

        w = np.linalg.solve(X_TX, X_Ty)

        self.intercept_ = w[0]

        self.coef_ = w[1:]

        return self

    def predict(self, X):

        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = self.intercept_ + X @ self.coef_

        return predictions

def example_simple_regression():

    print("ПРИМЕР 1. ПРОСТАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ. РЕАЛИЗАЦИЯ ЧЕРЕЗ КЛАСС")

    X = [
        [30],
        [40],
        [50]
    ]

    y = [
        5,
        6,
        8
    ]

    model = MyLinearRegression()

    model.fit(X, y)

    print("Свободный коэффициент w0:")
    print(model.intercept_)

    print("\nКоэффициент w1:")
    print(model.coef_[0])

    new_flat = [[60]]

    prediction = model.predict(new_flat)

    print("\nПрогноз стоимости квартиры площадью 60 м²:")
    print(prediction[0], "млн руб.")

def main():

    example_simple_regression()


if __name__ == "__main__":
    main()
