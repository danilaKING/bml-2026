import numpy as np


X = np.array([
    [1.0, 1.0],  # A
    [1.0, 2.0],  # B
    [2.0, 1.0],  # C
    [5.0, 5.0],  # D
    [6.0, 5.0],  # E
    [5.0, 6.0],  # F
])

names = np.array(["A", "B", "C", "D", "E", "F"])


centroids = np.array([
    [1.0, 1.0],
    [2.0, 1.0],
])


def assign_clusters(X, centroids):
    """
    Для каждой точки вычисляет расстояния
    до всех центроидов и возвращает индекс
    ближайшего центроида.
    """

    distances = np.linalg.norm(
        X[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )

    labels = np.argmin(distances, axis=1)

    return labels, distances


def update_centroids(X, labels, k):
    """
    Пересчитывает координаты центроидов
    как средние координаты объектов кластера.
    """

    new_centroids = np.array([
        X[labels == cluster].mean(axis=0)
        for cluster in range(k)
    ])

    return new_centroids


def calculate_inertia(X, labels, centroids):
    """
    Вычисляет сумму квадратов расстояний
    от точек до центроидов их кластеров.
    """

    inertia = 0

    for i in range(len(X)):
        centroid = centroids[labels[i]]
        inertia += np.sum((X[i] - centroid) ** 2)

    return inertia


K = 2

for iteration in range(10):

    print(f"\n--- Итерация {iteration + 1} ---")

    labels, distances = assign_clusters(X, centroids)

    for i, name in enumerate(names):
        print(
            f"{name}: "
            f"d1={distances[i, 0]:.2f}, "
            f"d2={distances[i, 1]:.2f}, "
            f"cluster={labels[i] + 1}"
        )

    new_centroids = update_centroids(X, labels, K)

    print("\nНовые центроиды:")
    print(new_centroids)

    inertia = calculate_inertia(X, labels, new_centroids)

    print(f"Inertia: {inertia:.2f}")

    # Если центры практически не изменились,
    # алгоритм сошелся.
    if np.allclose(centroids, new_centroids):
        print("\nАлгоритм сошелся.")
        break

    centroids = new_centroids


print("\nИтог:")

for cluster in range(K):
    print(
        f"Кластер {cluster + 1}:",
        names[labels == cluster].tolist()
    )