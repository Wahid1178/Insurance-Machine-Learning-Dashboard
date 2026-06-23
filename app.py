import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Insurance ML Dashboard",
    page_icon="🤖",
    layout="wide"
)

MODELS_DIR   = "models"
REG_SUBFOLDER = "regression"
CLF_SUBFOLDER = "classification"

REG_MODELS_DIR = os.path.join(MODELS_DIR, REG_SUBFOLDER)
CLF_MODELS_DIR = os.path.join(MODELS_DIR, CLF_SUBFOLDER)

VALID_MODEL_EXTENSIONS = (".pkl", ".joblib")


def discover_models_in_folder(folder_path):
    discovered = {}
    if not os.path.isdir(folder_path):
        return discovered
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(VALID_MODEL_EXTENSIONS):
            label = os.path.splitext(filename)[0]
            discovered[label] = os.path.join(folder_path, filename)
    return discovered

# =========================================================================
# CSS
# =========================================================================
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .big-title { font-size: 42px; font-weight: 800; color: white; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #C9D1D9; text-align: center; margin-bottom: 30px; }
    .card { background-color: #161B22; padding: 20px; border-radius: 16px; border: 1px solid #30363D; color: white; }
    .stButton > button { width: 100%; border-radius: 12px; height: 48px; font-weight: bold; background-color: #238636; color: white; border: none; }
    .stButton > button:hover { background-color: #2EA043; color: white; }
    .model-badge { display: inline-block; background-color: #1F6FEB; color: white; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# HEADER
# =========================================================================
st.markdown('<div class="big-title">🤖 Insurance Machine Learning Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Prediksi Harga Asuransi dan Status Perokok Berdasarkan Dataset Insurance</div>', unsafe_allow_html=True)

# =========================================================================
# SESSION STATE
# =========================================================================
for key in ["reg_models","clf_models","reg_result","clf_result",
            "reg_input","clf_input","dataset","bulk_reg_result","bulk_clf_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.reg_models is None: st.session_state.reg_models = {}
if st.session_state.clf_models is None: st.session_state.clf_models = {}

# =========================================================================
# FUNCTIONS — MODEL LOADING
# =========================================================================
def load_model_from_path(path):
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        return joblib.load(path)
    except Exception as e:
        st.sidebar.error(f"Gagal memuat {path}: {e}")
        return None

def load_model_from_upload(file):
    if file.name.endswith(".pkl"):
        return pickle.load(file)
    return joblib.load(file)

@st.cache_resource(show_spinner=False)
def load_all_embedded_models(reg_paths_tuple, clf_paths_tuple):
    reg_models = {}
    for label, path in reg_paths_tuple:
        model = load_model_from_path(path)
        if model is not None:
            reg_models[label] = model
    clf_models = {}
    for label, path in clf_paths_tuple:
        model = load_model_from_path(path)
        if model is not None:
            clf_models[label] = model
    return reg_models, clf_models

# =========================================================================
# FUNCTIONS — PREPROCESSING
# =========================================================================

REG_FEATURE_COLS = [
    "age","sex","bmi","children",
    "region_northeast","region_northwest","region_southeast","region_southwest",
    "smoker"
]
CLF_FEATURE_COLS = [
    "age","sex","bmi","children",
    "region_northeast","region_northwest","region_southeast","region_southwest"
]

# Mean & std dari insurance.csv setelah preprocessing (untuk scale input manual)
TRAIN_MEANS = {
    "age": 39.21, "sex": 0.505, "bmi": 30.67, "children": 1.094,
    "region_northeast": 0.242, "region_northwest": 0.242,
    "region_southeast": 0.272, "region_southwest": 0.243,
    "smoker": 0.205,
}
TRAIN_STDS = {
    "age": 14.05, "sex": 0.500, "bmi": 5.90, "children": 1.205,
    "region_northeast": 0.429, "region_northwest": 0.429,
    "region_southeast": 0.445, "region_southwest": 0.429,
    "smoker": 0.404,
}


def _encode_base(df):
    """Encode sex, smoker, region — berlaku untuk dataset maupun input manual."""
    df = df.copy()

    # sex
    if "sex" in df.columns:
        df["sex"] = df["sex"].apply(lambda x: 1 if str(x).lower() == "male" else 0)

    # smoker
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].apply(
            lambda x: 1 if str(x).lower() == "yes" else (0 if str(x).lower() == "no" else int(x))
        )

    # region → one-hot
    region_cols = ["region_northeast","region_northwest","region_southeast","region_southwest"]
    if "region" in df.columns:
        for col in region_cols:
            region_val = col.replace("region_", "")
            df[col] = (df["region"].str.lower() == region_val).astype(int)
        df.drop("region", axis=1, inplace=True)
    else:
        for col in region_cols:
            if col not in df.columns:
                df[col] = 0

    return df


def preprocess_dataset(df, task="regression"):
    df = df.copy()

    # 1. Hapus duplikat — SIMPAN df bersih untuk dikembalikan ke caller
    df = df.drop_duplicates().reset_index(drop=True)
    df_clean = df.copy()  # simpan versi bersih sebelum diubah lebih lanjut

    # 2. Outlier BMI
    if "bmi" in df.columns:
        m, s = df["bmi"].mean(), df["bmi"].std()
        if s > 0:
            df.loc[abs((df["bmi"] - m) / s) > 3, "bmi"] = m

    # 2b. Outlier charges
    if "charges" in df.columns:
        m, s = df["charges"].mean(), df["charges"].std()
        if s > 0:
            df.loc[abs((df["charges"] - m) / s) > 3, "charges"] = m

    # 3-5. Encode
    df = _encode_base(df)

    # Simpan target sebelum scaling
    y_charges = df["charges"].copy() if "charges" in df.columns else None
    y_smoker  = df["smoker"].copy()  if "smoker"  in df.columns else None

    # 6. Pilih fitur sesuai task
    if task == "regression":
        feat_cols = REG_FEATURE_COLS
    else:
        feat_cols = CLF_FEATURE_COLS

    # Pastikan semua kolom ada
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feat_cols].copy()

    # StandardScaler — fit dari data yang diupload (karena scaler tidak disimpan)
    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Kembalikan X, target, dan df_clean (untuk result_df di prediksi massal)
    if task == "regression":
        return X, y_charges, df_clean
    else:
        return X, y_smoker, df_clean


def preprocess_single_input(df, task="regression"):
    """
    Preprocessing untuk 1 baris input dari form manual.
    Tidak ada outlier handling. Scaling pakai mean/std dari data training.
    """
    df = _encode_base(df.copy())

    # Pilih fitur
    feat_cols = REG_FEATURE_COLS if task == "regression" else CLF_FEATURE_COLS

    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feat_cols].copy()

    # Scale pakai statistik training
    for col in X.columns:
        if col in TRAIN_MEANS and TRAIN_STDS.get(col, 0) > 0:
            X[col] = (X[col] - TRAIN_MEANS[col]) / TRAIN_STDS[col]

    return X


def smoker_label(pred):
    if pred in [1, "1", "yes", "Yes", "YES"]:
        return "Perokok"
    elif pred in [0, "0", "no", "No", "NO"]:
        return "Bukan Perokok"
    return str(pred)


# =========================================================================
# LOAD EMBEDDED MODELS
# =========================================================================
REG_MODEL_PATHS = discover_models_in_folder(REG_MODELS_DIR)
CLF_MODEL_PATHS = discover_models_in_folder(CLF_MODELS_DIR)

embedded_reg_models, embedded_clf_models = load_all_embedded_models(
    tuple(REG_MODEL_PATHS.items()),
    tuple(CLF_MODEL_PATHS.items())
)

for label, model in embedded_reg_models.items():
    if label not in st.session_state.reg_models:
        st.session_state.reg_models[label] = model

for label, model in embedded_clf_models.items():
    if label not in st.session_state.clf_models:
        st.session_state.clf_models[label] = model

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.header("⚙️ Status Model & Dataset")

st.sidebar.markdown("**📦 Model Regresi**")
if len(embedded_reg_models) == 0:
    st.sidebar.warning(f"Belum ada model di `{REG_MODELS_DIR}/`. Taruh file lalu refresh.")
else:
    for label in REG_MODEL_PATHS:
        st.sidebar.success(f"✅ {label}")

st.sidebar.markdown("**📦 Model Klasifikasi**")
if len(embedded_clf_models) == 0:
    st.sidebar.warning(f"Belum ada model di `{CLF_MODELS_DIR}/`. Taruh file lalu refresh.")
else:
    for label in CLF_MODEL_PATHS:
        st.sidebar.success(f"✅ {label}")

if st.sidebar.button("🔄 Refresh / Scan Ulang Folder Model"):
    st.cache_resource.clear()
    st.session_state.reg_models = {}
    st.session_state.clf_models = {}
    st.rerun()
st.sidebar.caption("Klik setiap kali menambah/mengganti file model di folder models/.")

st.sidebar.markdown("---")
st.sidebar.markdown("**📁 Upload Dataset**")
dataset_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])
if dataset_file is not None:
    try:
        st.session_state.dataset = pd.read_csv(dataset_file)
        st.sidebar.success("Dataset berhasil diupload")
    except Exception as e:
        st.sidebar.error(f"Gagal upload dataset: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset Semua Hasil Prediksi"):
    for k in ["reg_result","clf_result","reg_input","clf_input","bulk_reg_result","bulk_clf_result"]:
        st.session_state[k] = None
    st.success("Semua hasil prediksi berhasil direset.")

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Regresi Charges",
    "🚬 Klasifikasi Smoker",
    "📁 Dataset & Prediksi Massal",
    "📊 Evaluasi Model"
])

# =========================================================================
# TAB 1 — REGRESI (Prediksi Manual)
# =========================================================================
with tab1:
    st.subheader("💰 Prediksi Harga Asuransi / Charges")

    if len(st.session_state.reg_models) == 0:
        st.warning("Belum ada model regresi tersedia.")
    else:
        selected_reg_label = st.selectbox(
            "🔽 Pilih Model Regresi",
            list(st.session_state.reg_models.keys()),
            key="reg_model_select"
        )
        st.markdown(f'<div class="model-badge">Model aktif: {selected_reg_label}</div>', unsafe_allow_html=True)
        selected_reg_model = st.session_state.reg_models[selected_reg_label]

        col1, col2 = st.columns(2)
        with col1:
            age    = st.number_input("Age", 18, 100, 25, key="reg_age")
            sex    = st.selectbox("Sex", ["male","female"], key="reg_sex")
            bmi    = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1, key="reg_bmi")
        with col2:
            children = st.number_input("Children", 0, 10, 0, key="reg_children")
            smoker   = st.selectbox("Smoker", ["yes","no"], key="reg_smoker")
            region   = st.selectbox("Region", ["southwest","southeast","northwest","northeast"], key="reg_region")

        reg_input = pd.DataFrame({
            "age":[age],"sex":[sex],"bmi":[bmi],
            "children":[children],"smoker":[smoker],"region":[region]
        })

        st.write("### 🧾 Data Input Regresi")
        st.dataframe(reg_input, width='stretch')

        if st.button("🚀 Prediksi Charges"):
            try:
                X_reg = preprocess_single_input(reg_input, task="regression")
                pred  = selected_reg_model.predict(X_reg)[0]
                st.session_state.reg_result = float(pred)
                st.session_state.reg_input  = reg_input
                st.success(f"Prediksi berhasil menggunakan {selected_reg_label}.")
                st.metric("Prediksi Charges", f"{float(pred):,.2f}")
            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

        if st.session_state.reg_result is not None:
            prediction = st.session_state.reg_result
            colA, colB = st.columns(2)
            with colA:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction,
                    title={"text":"Prediksi Charges"},
                    gauge={
                        "axis":{"range":[0, max(prediction*1.5, 50000)]},
                        "bar":{"color":"#1F6FEB"},
                        "steps":[
                            {"range":[0,10000],"color":"#D0E7FF"},
                            {"range":[10000,30000],"color":"#79C0FF"},
                            {"range":[30000,60000],"color":"#388BFD"}
                        ]
                    }
                ))
                fig_gauge.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_gauge, width='stretch')
            with colB:
                summary_df = pd.DataFrame({
                    "Fitur":["Age","BMI","Children"],
                    "Nilai":[
                        st.session_state.reg_input["age"].iloc[0],
                        st.session_state.reg_input["bmi"].iloc[0],
                        st.session_state.reg_input["children"].iloc[0]
                    ]
                })
                fig_bar = px.bar(summary_df, x="Fitur", y="Nilai", text="Nilai",
                                 title="Ringkasan Input Numerik", template="plotly_dark")
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, width='stretch')

# =========================================================================
# TAB 2 — KLASIFIKASI (Prediksi Manual)
# =========================================================================
with tab2:
    st.subheader("🚬 Prediksi Status Smoker")

    if len(st.session_state.clf_models) == 0:
        st.warning("Belum ada model klasifikasi tersedia.")
    else:
        selected_clf_label = st.selectbox(
            "🔽 Pilih Model Klasifikasi",
            list(st.session_state.clf_models.keys()),
            key="clf_model_select"
        )
        st.markdown(f'<div class="model-badge">Model aktif: {selected_clf_label}</div>', unsafe_allow_html=True)
        selected_clf_model = st.session_state.clf_models[selected_clf_label]

        col1, col2 = st.columns(2)
        with col1:
            age  = st.number_input("Age", 18, 100, 25, key="clf_age")
            sex  = st.selectbox("Sex", ["male","female"], key="clf_sex")
            bmi  = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1, key="clf_bmi")
        with col2:
            children = st.number_input("Children", 0, 10, 0, key="clf_children")
            region   = st.selectbox("Region", ["southwest","southeast","northwest","northeast"], key="clf_region")

        clf_input = pd.DataFrame({
            "age":[age],"sex":[sex],"bmi":[bmi],
            "children":[children],"region":[region]
        })

        st.write("### 🧾 Data Input Klasifikasi")
        st.dataframe(clf_input, width='stretch')

        if st.button("🚀 Prediksi Smoker"):
            try:
                X_clf = preprocess_single_input(clf_input, task="classification")
                pred  = selected_clf_model.predict(X_clf)[0]
                hasil = smoker_label(pred)
                st.session_state.clf_result = hasil
                st.session_state.clf_input  = clf_input
                st.success(f"Prediksi berhasil menggunakan {selected_clf_label}.")
                st.metric("Prediksi Status Smoker", hasil)
            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

        if st.session_state.clf_result is not None:
            hasil = st.session_state.clf_result
            smoker_df = pd.DataFrame({
                "Status":["Perokok","Bukan Perokok"],
                "Nilai":[1 if hasil=="Perokok" else 0, 1 if hasil=="Bukan Perokok" else 0]
            })
            colA, colB = st.columns(2)
            with colA:
                fig_pie = px.pie(smoker_df, names="Status", values="Nilai", hole=0.45,
                                 title="Visualisasi Hasil Prediksi Smoker", template="plotly_dark")
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, width='stretch')
            with colB:
                input_summary = pd.DataFrame({
                    "Fitur":["Age","BMI","Children"],
                    "Nilai":[
                        st.session_state.clf_input["age"].iloc[0],
                        st.session_state.clf_input["bmi"].iloc[0],
                        st.session_state.clf_input["children"].iloc[0]
                    ]
                })
                fig_bar = px.bar(input_summary, x="Fitur", y="Nilai", text="Nilai",
                                 title="Ringkasan Input Klasifikasi", template="plotly_dark")
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, width='stretch')

# =========================================================================
# TAB 3 — DATASET & PREDIKSI MASSAL
# =========================================================================
with tab3:
    st.subheader("📁 Dataset Insurance")

    if st.session_state.dataset is None:
        st.info("Upload dataset CSV terlebih dahulu di sidebar.")
    else:
        df = st.session_state.dataset

        st.write("### Preview Dataset (Keseluruhan Data)")
        st.dataframe(df, width='stretch', height=350)

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df.shape[0])
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Missing Value", int(df.isnull().sum().sum()))

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            if "charges" in df.columns:
                st.plotly_chart(px.histogram(df, x="charges", nbins=30,
                    title="Distribusi Charges", template="plotly_dark"), width='stretch')
        with colB:
            if "smoker" in df.columns:
                sc = df["smoker"].value_counts().reset_index()
                sc.columns = ["smoker","jumlah"]
                st.plotly_chart(px.pie(sc, names="smoker", values="jumlah",
                    title="Distribusi Smoker", hole=0.45, template="plotly_dark"), width='stretch')

        colC, colD = st.columns(2)
        with colC:
            if "region" in df.columns:
                rc = df["region"].value_counts().reset_index()
                rc.columns = ["region","jumlah"]
                fig_r = px.bar(rc, x="region", y="jumlah", text="jumlah",
                               title="Jumlah Data Berdasarkan Region", template="plotly_dark")
                fig_r.update_traces(textposition="outside")
                st.plotly_chart(fig_r, width='stretch')
        with colD:
            if "age" in df.columns and "charges" in df.columns:
                st.plotly_chart(px.scatter(df, x="age", y="charges",
                    color="smoker" if "smoker" in df.columns else None,
                    title="Hubungan Age dan Charges", template="plotly_dark"), width='stretch')

        st.markdown("---")
        st.write("## 🚀 Prediksi Massal (Seluruh Baris Dataset)")
        st.caption("Pilih model lalu jalankan prediksi untuk seluruh baris dataset.")

        pred_task = st.radio("Pilih Jenis Prediksi",
            ["Prediksi Charges (Regresi)","Prediksi Smoker (Klasifikasi)"],
            horizontal=True, key="bulk_pred_task")

        if pred_task == "Prediksi Charges (Regresi)":
            if len(st.session_state.reg_models) == 0:
                st.warning("Belum ada model regresi tersedia.")
            else:
                bulk_reg_label = st.selectbox("🔽 Pilih Model Regresi",
                    list(st.session_state.reg_models.keys()), key="bulk_reg_model_select")

                if st.button("🚀 Jalankan Prediksi Charges untuk Seluruh Data"):
                    try:
                        model = st.session_state.reg_models[bulk_reg_label]
                        X_ready, _, df_clean = preprocess_dataset(df, task="regression")
                        preds = model.predict(X_ready)
                        result_df = df_clean.copy()
                        result_df["Predicted_Charges"] = preds
                        st.session_state.bulk_reg_result = result_df
                        st.success(f"Prediksi massal berhasil menggunakan {bulk_reg_label}.")
                    except Exception as e:
                        st.error(f"Prediksi massal gagal: {e}")

                if st.session_state.bulk_reg_result is not None:
                    st.write("### 📋 Hasil Prediksi Charges")
                    st.dataframe(st.session_state.bulk_reg_result, width='stretch', height=350)
                    st.download_button("⬇️ Download CSV",
                        data=st.session_state.bulk_reg_result.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi_charges.csv", mime="text/csv")
        else:
            if len(st.session_state.clf_models) == 0:
                st.warning("Belum ada model klasifikasi tersedia.")
            else:
                bulk_clf_label = st.selectbox("🔽 Pilih Model Klasifikasi",
                    list(st.session_state.clf_models.keys()), key="bulk_clf_model_select")

                if st.button("🚀 Jalankan Prediksi Smoker untuk Seluruh Data"):
                    try:
                        model = st.session_state.clf_models[bulk_clf_label]
                        X_ready, _, df_clean = preprocess_dataset(df, task="classification")
                        preds = model.predict(X_ready)
                        result_df = df_clean.copy()
                        result_df["Predicted_Smoker"] = [smoker_label(p) for p in preds]
                        st.session_state.bulk_clf_result = result_df
                        st.success(f"Prediksi massal berhasil menggunakan {bulk_clf_label}.")
                    except Exception as e:
                        st.error(f"Prediksi massal gagal: {e}")

                if st.session_state.bulk_clf_result is not None:
                    st.write("### 📋 Hasil Prediksi Smoker")
                    st.dataframe(st.session_state.bulk_clf_result, width='stretch', height=350)
                    st.download_button("⬇️ Download CSV",
                        data=st.session_state.bulk_clf_result.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi_smoker.csv", mime="text/csv")

# =========================================================================
# TAB 4 — EVALUASI MODEL
# =========================================================================
with tab4:
    st.subheader("📊 Evaluasi Model")

    if st.session_state.dataset is None:
        st.info("Upload dataset CSV terlebih dahulu untuk evaluasi.")
    else:
        df_eval = st.session_state.dataset.copy()
        st.write("Dataset tersedia untuk evaluasi.")

        eval_type = st.selectbox("Pilih Evaluasi",
            ["Evaluasi Regresi Charges","Evaluasi Klasifikasi Smoker"])

        # ------------------------------------------------------------------
        # EVALUASI REGRESI
        # ------------------------------------------------------------------
        if eval_type == "Evaluasi Regresi Charges":
            if len(st.session_state.reg_models) == 0:
                st.warning("Belum ada model regresi tersedia.")
            elif "charges" not in df_eval.columns:
                st.error("Kolom 'charges' tidak ditemukan pada dataset.")
            else:
                eval_reg_label = st.selectbox("🔽 Pilih Model Regresi",
                    list(st.session_state.reg_models.keys()), key="eval_reg_model_select")
                try:
                    model = st.session_state.reg_models[eval_reg_label]

                    # Preprocessing: fitur [age,sex,bmi,children,region_*,smoker], target=charges
                    X_ready, y, _ = preprocess_dataset(df_eval, task="regression")
                    y_pred = model.predict(X_ready)

                    r2   = r2_score(y, y_pred)
                    mae  = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))

                    st.markdown(f'<div class="model-badge">Model dievaluasi: {eval_reg_label}</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{r2:.4f}")
                    col2.metric("MAE", f"{mae:,.2f}")
                    col3.metric("RMSE", f"{rmse:,.2f}")

                    fig_eval = px.scatter(
                        pd.DataFrame({"Actual":y,"Predicted":y_pred}),
                        x="Actual", y="Predicted",
                        title=f"Actual vs Predicted Charges — {eval_reg_label}",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_eval, width='stretch')

                except Exception as e:
                    st.error(f"Evaluasi regresi gagal: {e}")

        # ------------------------------------------------------------------
        # EVALUASI KLASIFIKASI
        # ------------------------------------------------------------------
        else:
            if len(st.session_state.clf_models) == 0:
                st.warning("Belum ada model klasifikasi tersedia.")
            elif "smoker" not in df_eval.columns:
                st.error("Kolom 'smoker' tidak ditemukan pada dataset.")
            else:
                eval_clf_label = st.selectbox("🔽 Pilih Model Klasifikasi",
                    list(st.session_state.clf_models.keys()), key="eval_clf_model_select")
                try:
                    model = st.session_state.clf_models[eval_clf_label]

                    # Preprocessing: fitur [age,sex,bmi,children,region_*], target=smoker
                    X_ready, y_raw, _ = preprocess_dataset(df_eval, task="classification")

                    # Pastikan y bertipe int (0/1)
                    y = y_raw.apply(
                        lambda x: int(x) if str(x).isdigit()
                        else (1 if str(x).lower() == "yes" else 0)
                    )

                    y_pred = [int(p) for p in model.predict(X_ready)]

                    acc  = accuracy_score(y, y_pred)
                    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
                    rec  = recall_score(y, y_pred, average="weighted", zero_division=0)
                    f1   = f1_score(y, y_pred, average="weighted", zero_division=0)

                    st.markdown(f'<div class="model-badge">Model dievaluasi: {eval_clf_label}</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy",  f"{acc:.2%}")
                    col2.metric("Precision", f"{prec:.2%}")
                    col3.metric("Recall",    f"{rec:.2%}")
                    col4.metric("F1 Score",  f"{f1:.2%}")

                    labels = sorted(y.unique())
                    cm     = confusion_matrix(y, y_pred, labels=labels)
                    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)

                    st.plotly_chart(px.imshow(cm_df, text_auto=True,
                        title=f"Confusion Matrix — {eval_clf_label}",
                        template="plotly_dark"), width='stretch')

                    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                    st.write("### Classification Report")
                    st.dataframe(pd.DataFrame(report).transpose(), width='stretch')

                except Exception as e:
                    st.error(f"Evaluasi klasifikasi gagal: {e}")

st.markdown("---")
st.caption("© 2026 Insurance Machine Learning Dashboard")