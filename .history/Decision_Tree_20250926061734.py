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

# pilih fitur relevan (tanpa G3 agar tidak bocor)
X = data.iloc[:, [5, 6, 7, 8, 9]].values   # studytime, failures, absences, G1, G2
y = data.iloc[:, -1].values                # target Pass/Fail

print("Data Variabel".center(75, "="))
print(X[:10])   # tampilkan 10 baris
print("Data Kelas".center(75, "="))
print(y[:10])
print("="*75, "\n")

# ubah target jadi numerik
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
# 5. Pemodelan Decision Tree
# ====================================================
print("PEMODELAN DECISION TREE".center(75, "="))
decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
decision_tree.fit(X_train, y_train)
print("Model Decision Tree sudah terbentuk.")
print("="*75, "\n")

# ====================================================
# 6. Prediksi Decision Tree
# ====================================================
print("PREDIKSI DECISION TREE".center(75, "="))
Y_pred = decision_tree.predict(X_test)
print("Hasil Prediksi (20 pertama):", Y_pred[:20])
print("="*75, "\n")

# ====================================================
# 7. Prediksi Akurasi
# ====================================================
print("PREDIKSI AKURASI".center(75, "="))
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

# ====================================================
# 10. Contoh Input Manual
# ====================================================
print("CONTOH INPUT MANUAL".center(75, "="))
studytime = int(input("Masukkan studytime (1=kurang, 2=sedang, 3=bagus, 4=sangat bagus): "))
failures  = int(input("Masukkan jumlah kegagalan (0-3): "))
absences  = int(input("Masukkan jumlah ketidakhadiran: "))
G1        = int(input("Masukkan nilai ujian pertama (0-20): "))
G2        = int(input("Masukkan nilai ujian kedua (0-20): "))

# buat dataframe untuk prediksi
sample = pd.DataFrame([[studytime, failures, absences, G1, G2]],
                      columns=["studytime","failures","absences","G1","G2"])

prediksi = decision_tree.predict(sample)[0]
print("Hasil Prediksi:", "Pass" if prediksi==1 else "Fail")
print("="*75, "\n")
