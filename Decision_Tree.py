# ====================================================
# 1. Import Library
# ====================================================
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

# fitur yang dipakai (hindari G3 agar tidak bocor)
fitur = ["studytime", "failures", "absences", "G1", "G2"]
X = data[fitur]             # DataFrame dengan feature names
y = data.iloc[:, -1]        # target terakhir (Pass/Fail)

print("Data Variabel".center(75, "="))
print(X.head(10))           # tampilkan 10 baris
print("Data Kelas".center(75, "="))
print(y.head(10))
print("="*75, "\n")

# ubah target jadi numerik
if y.dtype == object:
    y = y.str.lower().map(lambda v: 1 if str(v).strip() in ("pass","yes","1","true") else 0)

# ====================================================
# 4. Training dan Testing
# ====================================================
print("SPLITTING DATA 80-20".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Instance variabel data training".center(75, "="))
print(X_train.head())
print("Instance kelas data training".center(75, "="))
print(y_train.head())
print("Instance variabel data testing".center(75, "="))
print(X_test.head())
print("Instance kelas data testing".center(75, "="))
print(y_test.head())
print("="*75, "\n")

# ====================================================
# 5. Pemodelan Decision Tree
# ====================================================
print("PEMODELAN DECISION TREE".center(75, "="))
decision_tree = DecisionTreeClassifier(
    criterion="entropy",  # pakai entropy biar sesuai modul
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
print("Hasil Prediksi (20 pertama):")
print(Y_pred[:20])
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
    feature_names=fitur,
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
print("COBA INPUT".center(75, "="))

A = int(input("Masukkan studytime (1=<2h, 2=2-5h, 3=5-10h, 4=>10h per minggu): "))
B = int(input("Masukkan jumlah kegagalan (0-3): "))
C = int(input("Masukkan jumlah ketidakhadiran (0-93): "))
D = int(input("Masukkan nilai ujian pertama G1 (0-20): "))
E = int(input("Masukkan nilai ujian kedua G2 (0-20): "))

# bentuk DataFrame sesuai fitur
sample = pd.DataFrame([[A, B, C, D, E]], columns=fitur)
print("Data input:")
print(sample)

prediksi = decision_tree.predict(sample)[0]
if prediksi == 1:
    print("Hasil Prediksi: Siswa PASS ✅")
else:
    print("Hasil Prediksi: Siswa FAIL ❌")

print("="*75, "\n")
