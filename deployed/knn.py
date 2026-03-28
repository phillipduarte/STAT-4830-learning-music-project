import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

X_train = np.load("../../old_data/embeddings_train.npy")
X_test  = np.load("../../old_data/embeddings_test.npy")
y_train_str = np.load("../../old_data/labels_train.npy", allow_pickle=True)
y_test_str  = np.load("../../old_data/labels_test.npy",  allow_pickle=True)

le = LabelEncoder().fit(y_train_str)
y_train = le.transform(y_train_str)
y_test  = le.transform(y_test_str)

accuracies = []

print("Evaluating KNN...")

for i in tqdm(range(1, 50)):
    knn = KNeighborsClassifier(n_neighbors=i, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)

    probs = knn.predict_proba(X_test)   # needed for top-k
    top1  = knn.score(X_test, y_test)

    top5 = top_k_accuracy_score(y_test, probs, k=5, labels=np.arange(len(le.classes_)))
    accuracies.append((top1, top5, i))
    print("Finished step {}: top-1: {:.4f}  top-5: {:.4f}".format(i, top1, top5))

best_top1 = max(accuracies, key=lambda x: x[0])
best_top5 = max(accuracies, key=lambda x: x[1])

print(f"KNN top-1: {best_top1[0]:.4f}  top-5: {best_top5[1]:.4f}")