import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42
)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    alpha=0.7
)

plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.title("Исходные данные")

plt.tight_layout()
plt.savefig("01_data.png", dpi=200)
plt.show()


k_values = range(1, 11)
inertias = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)


plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Количество кластеров K")
plt.ylabel("Inertia")
plt.title("Метод локтя")

plt.xticks(list(k_values))

plt.tight_layout()
plt.savefig("02_elbow.png", dpi=200)
plt.show()


silhouette_values = []

k_values_silhouette = range(2, 11)

for k in k_values_silhouette:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)

    print(
        f"K={k}: "
        f"silhouette={score:.3f}"
    )


plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values_silhouette),
    silhouette_values,
    marker="o"
)

plt.xlabel("Количество кластеров K")
plt.ylabel("Silhouette score")
plt.title("Выбор K по silhouette score")

plt.tight_layout()
plt.savefig("03_silhouette.png", dpi=200)
plt.show()


K = 4

model = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    random_state=42
)

labels = model.fit_predict(X_scaled)

centroids = model.cluster_centers_


plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    alpha=0.7
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    edgecolors="black",
    linewidths=2,
    label="Центроиды"
)

plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.title("Результат кластеризации K-means")

plt.legend()

plt.tight_layout()
plt.savefig("04_kmeans_result.png", dpi=200)
plt.show()


print("\nЦентроиды:")
print(model.cluster_centers_)

print("\nInertia:")
print(model.inertia_)

print("\nКоличество итераций:")
print(model.n_iter_)

print("\nSilhouette score:")
print(
    silhouette_score(
        X_scaled,
        labels
    )
)
