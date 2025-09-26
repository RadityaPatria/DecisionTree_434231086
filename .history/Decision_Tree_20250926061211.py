# ====================================================
# 1. Import Library
# ====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ====================================================
# 2. Load Data
# ====================================================
data = pd.read_csv("student-mat-pass-or-fail.csv")

print("DATA AWAL".center(75, "="))
print(data.head())
print("="*75, "\n")

# ====================================================
# 3. Grouping Variabel & Kelas
# ====================================================
print("GROUPING VARIABEL".center(75, "="))

# Pilih kolom variabel (contoh: ambil kolom ke-5 s/d ke-9) -> studytime, failures, absences, G1, G2
X = data.iloc[:, [5, 6, 7, 8, 9]].values
y = data.iloc[:, -1].values   # target di kolom terakhir

print("Data Variabel".center(75, "="))
print(X[:10])   # print 10 baris pertama saja biar rapi
print("Data Kelas".center(75, "="))
print(y[:10])
print("="*75, "\n")

# pastikan target numerik (0=Fail, 1=Pass)
y = pd.Series(y)
if y.dtype == object:
    y = y.str.lower().map(lambda v: 1 if str(v).strip() in ("pass","yes","1","true") else 0).values

# ====================================================
# 4. Training dan Testing
# ====================================================
print("SPLITTING DATA 80-20".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Instance variabel data training".center(75, "="))
print(X_train[:5])
print("Instance kelas data training".center(75, "="))
print(y_train[:5])
print("Instance variabel data testing".center(75, "="))
print(X_test[:5])
print("Instance kelas data testing".center(75, "="))
print(y_test[:5])
print("="*75, "\n")

# ====================================================
# 5. Pemodelan Decision Tree (pakai entropy)
# ====================================================
decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
decision_tree.fit(X_train, y_train)

# ====================================================
# 6. Prediksi Decision Tree
# ====================================================
print("INSTANCE PREDIKSI DECISION TREE".center(75, "="))
Y_pred = decision_tree.predict(X_test)
print(Y_pred[:20])  # tampilkan 20 prediksi pertama
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
plt.figure(figsize=(25,15))
plot_tree(
    decision_tree,
    feature_names=["studytime", "failures", "absences", "G1", "G2"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Visualisasi Decision Tree (Entropy)")
plt.show()
