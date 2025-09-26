# ====================================================
# 1. Import Library
# ====================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ====================================================
# 2. Menampilkan Semua Data
# ====================================================
data = pd.read_csv("student-mat-pass-or-fail.csv")

print("DATA AWAL".center(75, "="))
print(data.head())
print("="*75, "\n")

print("INFO DATASET".center(75, "="))
print(data.info())
print("="*75, "\n")

print("DESKRIPSI DATASET".center(75, "="))
print(data.describe())
print("="*75, "\n")

# ====================================================
# 3. Grouping Variabel & Kelas
# ====================================================
print("GROUPING VARIABEL".center(75, "="))
X = data.iloc[:, :-1].values    # semua kolom kecuali terakhir (fitur)
y = data.iloc[:, -1].values     # kolom terakhir (kelas/target)

print("Data Variabel".center(75, "="))
print(X)
print("Data Kelas".center(75, "="))
print(y)
print("="*75, "\n")

# pastikan target dalam bentuk numerik
y = pd.Series(y)
if y.dtype == object:
    y = y.str.lower().map(lambda v: 1 if str(v).strip() in ("pass","yes","1","true") else 0).values

# ====================================================
# 4. Training dan Testing
# ====================================================
print("SPLITTING DATA 80-20".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("Instance variabel data training".center(75, "="))
print(X_train)
print("Instance kelas data training".center(75, "="))
print(y_train)
print("Instance variabel data testing".center(75, "="))
print(X_test)
print("Instance kelas data testing".center(75, "="))
print(y_test)
print("="*75, "\n")

# ====================================================
# 5. Pemodelan Decision Tree
# ====================================================
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

# ====================================================
# 6. Prediksi Decision Tree
# ====================================================
print("INSTANCE PREDIKSI DECISION TREE".center(75, "="))
Y_pred = decision_tree.predict(X_test)
print(Y_pred)
print("="*75, "\n")

# ====================================================
# 7. Prediksi Akurasi
# ====================================================
accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
print("Akurasi:", accuracy, "%")
print("="*75, "\n")

# ====================================================
# 8. Classification Report & Confusion Matrix
# ====================================================
print("CLASSIFICATION REPORT DECISION TREE".center(75, "="))
print(classification_report(y_test, Y_pred))

cm = confusion_matrix(y_test, Y_pred)
print("Confusion Matrix:")
print(cm)
print("="*75, "\n")

# ====================================================
# 9. Visualisasi Decision Tree
# ====================================================
plt.figure(figsize=(30,15))
plot_tree(
    decision_tree,
    feature_names=data.columns[:-1],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Visualisasi Decision Tree")
plt.show()

# ====================================================
# 10. Contoh Input Manual
# ====================================================
print("CONTOH INPUT MANUAL".center(75, "="))
# ambil 1 data dari testing
sample = X_test[0].reshape(1, -1)
print("Data uji:", sample)
prediksi = decision_tree.predict(sample)
print("Hasil Prediksi:", "Pass" if prediksi[0] == 1 else "Fail")
print("="*75, "\n")
