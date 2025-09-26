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
print(data.head(), "\n")

# ====================================================
# 3. Pilih fitur dan target
# ====================================================
# buang G3 biar tidak bocor
fitur = ["studytime", "failures", "absences", "G1", "G2"]
X = data[fitur].values
y = data.iloc[:, -1].values  # target terakhir (Pass/Fail)

# pastikan target numerik (0=Fail, 1=Pass)
y = pd.Series(y)
if y.dtype == object:
    y = y.str.lower().map(lambda v: 1 if str(v).strip() in ("pass","yes","1","true") else 0).values

print("FITUR YANG DIPAKAI".center(75, "="))
print(fitur)
print("="*75, "\n")

# ====================================================
# 4. Split Data
# ====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================================
# 5. Train Decision Tree
# ====================================================
decision_tree = DecisionTreeClassifier(
    criterion="gini",   # bisa "entropy" juga
    max_depth=5,        # batasi kedalaman biar kelihatan rapi
    random_state=42
)
decision_tree.fit(X_train, y_train)

# ====================================================
# 6. Prediksi & Evaluasi
# ====================================================
y_pred = decision_tree.predict(X_test)
print("Akurasi:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

# ====================================================
# 7. Visualisasi Pohon Keputusan
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
plt.title("Visualisasi Decision Tree dengan fitur yang relevan")
plt.show()
