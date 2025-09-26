import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================
# 1. Load Data
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("student-mat-pass-or-fail.csv")

data = load_data()

# =====================================
# 2. Title & Preview Data
# =====================================
st.title("🎓 Decision Tree Classifier - Student Performance")
st.write("Dataset: **Student Math Pass/Fail**")

if st.checkbox("Tampilkan data awal"):
    st.dataframe(data.head())

# =====================================
# 3. Grouping
# =====================================
fitur = ["studytime", "failures", "absences", "G1", "G2"]
X = data[fitur]
y = data.iloc[:, -1]

if y.dtype == object:
    y = y.str.lower().map(lambda v: 1 if str(v).strip() in ("pass","yes","1","true") else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 4. Model Training
# =====================================
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# =====================================
# 5. Evaluation
# =====================================
st.subheader("📊 Evaluasi Model")
st.write("**Akurasi:**", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

if st.checkbox("Tampilkan Classification Report"):
    st.text(classification_report(y_test, y_pred))

if st.checkbox("Tampilkan Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

# =====================================
# 6. Visualisasi Pohon Keputusan
# =====================================
st.subheader("🌳 Visualisasi Decision Tree")
fig, ax = plt.subplots(figsize=(20,10))
plot_tree(
    clf,
    feature_names=fitur,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
st.pyplot(fig)

# =====================================
# 7. Prediksi Input Manual (UI Form)
# =====================================
st.subheader("📝 Coba Prediksi Siswa Baru")

with st.form("student_form"):
    studytime = st.slider("Studytime (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)", 1, 4, 2)
    failures = st.slider("Jumlah Kegagalan (0-3)", 0, 3, 0)
    absences = st.slider("Jumlah Ketidakhadiran (0-93)", 0, 93, 5)
    G1 = st.slider("Nilai Ujian Pertama G1 (0-20)", 0, 20, 10)
    G2 = st.slider("Nilai Ujian Kedua G2 (0-20)", 0, 20, 10)
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    sample = pd.DataFrame([[studytime, failures, absences, G1, G2]], columns=fitur)
    pred = clf.predict(sample)[0]
    if pred == 1:
        st.success("✅ Hasil Prediksi: **PASS**")
    else:
        st.error("❌ Hasil Prediksi: **FAIL**")
