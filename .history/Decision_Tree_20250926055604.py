# ========================================
# DECISION TREE
# ========================================
# Nama File: Decision_Tree.py
# ----------------------------------------
# Urutan sesuai modul:
# 1. Import Library
# 2. Menampilkan Semua Data
# 3. Grouping Data
# 4. Training dan Testing
# 5. Decision Tree
# 6. Prediksi & Akurasi
# 7. Visualisasi Decision Tree
# 8. Klasifikasi / Prediksi Input
# ========================================

# 1. Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ----------------------------------------
# 2. Menampilkan Semua Data
# ----------------------------------------
df = pd.read_csv("student-mat-pass-or-fail.csv")  # pastikan file CSV ada di folder sama
print("=== Menampilkan 5 data teratas ===")
print(df.head())

print("\n=== Info Dataset ===")
print(df.info())

print("\n=== Deskripsi Dataset ===")
print(df.describe())

# ----------------------------------------
# 3. Grouping Data (contoh: jumlah Pass vs Fail)
# ----------------------------------------
print("\n=== Grouping berdasarkan kolom target (Pass/Fail) ===")
print(df.groupby("pass").size())

# ----------------------------------------
# 4. Training dan Testing
# ----------------------------------------
X = df.drop(columns=["pass"])   # fitur
y = df["pass"]                  # target

# Jika target string, ubah ke biner
if y.dtype == object:
    y = y.str.lower().map(lambda v: 1 if str(v).strip() in ("pass","yes","1","true") else 0)

# One-hot encoding untuk fitur kategorikal
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nJumlah data latih:", X_train.shape[0])
print("Jumlah data uji:", X_test.shape[0])

# ----------------------------------------
# 5. Decision Tree
# ----------------------------------------
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Simpan model
joblib.dump(clf, "decision_tree_model.joblib")

# ----------------------------------------
# 6. Prediksi & Akurasi
# ----------------------------------------
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n=== Akurasi Model ===")
print("Akurasi:", acc)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------
# 7. Visualisasi Decision Tree
# ----------------------------------------
plt.figure(figsize=(20,10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)
plt.title("Visualisasi Decision Tree")
plt.show()

# ----------------------------------------
# 8. Klasifikasi / Prediksi Input
# ----------------------------------------
print("\n=== Prediksi Data Input Manual ===")
# contoh input: ambil 1 baris dari data uji
sample = X_test.iloc[0:1]
print("Data uji:", sample)
prediksi = clf.predict(sample)
print("Hasil Prediksi:", "Pass" if prediksi[0] == 1 else "Fail")
