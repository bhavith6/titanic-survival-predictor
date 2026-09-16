import numpy as np
from collections import Counter
X_train = np.array([
    [1, 2], [2, 3], [3, 1],  
    [8, 9], [9, 8], [9, 10]  
])
y_train = np.array([0, 0, 0, 1, 1, 1])
test_point = np.array([2.45, 2.25])
def predict_knn(X_train, y_train, point, k=3):
    # Step 1: Vectorized distance calculation
    distances = np.sqrt(np.sum((X_train - point)**2, axis=1))
    print(f"Distances from {point} to all training points:")
    print(np.round(distances, 2))
    k_indices = np.argsort(distances)[:k]
    print(f"\nIndices of the {k} closest points: {k_indices}")
    k_labels = [y_train[i] for i in k_indices]
    print(f"Labels of those closest points: {k_labels}")
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]
prediction = predict_knn(X_train, y_train, test_point, k=3)
print(f"\nFINAL PREDICTION: Class {prediction}")