# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# -------- CONFIG -------
DATA_PATH = "housing.csv"
MODEL_PATH = "housing_model.joblib"
TARGET = "median_house_value"   # this dataset column will be predicted
# -----------------------

st.set_page_config(page_title="House Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("🏠 Real-time House Price Predictor")
st.markdown(
    "Enter property details on the left. The app trains a model from `housing.csv` (first run) "
    "and provides instant predictions and metrics."
)

# ---------- Helpers ----------
@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f"Can't find {path}. Place your CSV in the same folder as this app.")
        st.stop()
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        st.error(f"Target column '{TARGET}' not found in CSV. Columns: {list(df.columns)}")
        st.stop()
    return df

def build_and_train(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # --- SAFE computation of RMSE for all sklearn versions ---
    mse = mean_squared_error(y_test, y_pred)      # mean squared error
    rmse = mse ** 0.5                             # root mean squared error
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }

    return pipe, metrics, X_train, X_test, y_train, y_test


def load_or_train(df):
    if os.path.exists(MODEL_PATH):
        try:
            pipe, metrics = joblib.load(MODEL_PATH)
            return pipe, metrics
        except Exception:
            # if loading fails, retrain
            pass
    pipe, metrics, _, _, _, _ = build_and_train(df)
    joblib.dump((pipe, metrics), MODEL_PATH)
    return pipe, metrics

# ---------- Prepare data + model ----------
df = load_data()
pipe, metrics = load_or_train(df)

# ---------- Sidebar: user inputs ----------
st.sidebar.header("Enter property details")

# We'll pick the most useful fields from dataset; if you have different columns adapt them here
# Required columns (most common in the dataset): 
expected_inputs = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income", "ocean_proximity"
]

# Safety: only show controls for columns the CSV actually contains
available = df.columns.tolist()
controls = {}
for col in expected_inputs:
    if col not in available:
        continue
    if df[col].dtype.kind in "biuf":  # numeric
        minv = float(df[col].min())
        maxv = float(df[col].max())
        med = float(df[col].median())
        if col in ["longitude", "latitude"]:
            controls[col] = st.sidebar.slider(col, min_value=minv, max_value=maxv, value=med, format="%.4f")
        elif col == "median_income":
            controls[col] = st.sidebar.slider(col, min_value=round(minv,1), max_value=round(maxv,1), value=round(med,2), step=0.1)
        else:
            # for large integers use number_input
            if maxv - minv > 50:
                controls[col] = st.sidebar.number_input(col, min_value=int(minv), max_value=int(maxv), value=int(med), step=1)
            else:
                controls[col] = st.sidebar.slider(col, min_value=int(minv), max_value=int(maxv), value=int(med), step=1)
    else:
        # categorical
        choices = df[col].dropna().unique().tolist()
        controls[col] = st.sidebar.selectbox(col, choices)

# Predict button
predict_btn = st.sidebar.button("Predict")

# ---------- Main display ----------
col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Model performance")
    st.metric("R² (accuracy)", f"{metrics['r2'] * 100:.2f}%")
    st.metric("RMSE", f"${metrics['rmse']:,.0f}")
    st.metric("MAE", f"${metrics['mae']:,.0f}")

    st.markdown("**Dataset snapshot**")
    st.dataframe(df.head(6))

with col_right:
    st.subheader("Prediction")

    if predict_btn:
        # build a one-row DataFrame with same columns as training X
        X_example = df.drop(columns=[TARGET]).iloc[:1].copy()  # use structure
        for c in X_example.columns:
            if c in controls:
                X_example.at[X_example.index[0], c] = controls[c]
            else:
                # if control not provided, use median
                if df[c].dtype.kind in "biuf":
                    X_example.at[X_example.index[0], c] = df[c].median()
                else:
                    X_example.at[X_example.index[0], c] = df[c].mode().iloc[0]
        # predict
        pred = pipe.predict(X_example)[0]
        st.markdown("<div style='font-size:34px; font-weight:700;'>Estimated price</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:40px; color:green; font-weight:800;'>${pred:,.0f}</div>", unsafe_allow_html=True)
        st.success("Prediction completed")
        st.balloons()

        # show prediction on histogram
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df[TARGET], bins=50)
        ax.axvline(pred, linewidth=2)
        ax.set_xlabel("Median house value")
        ax.set_ylabel("Count")
        ax.set_title("Price distribution (red line = prediction)")
        st.pyplot(fig)

         # show 5 similar houses (by simple nearest neighbors on numeric features)
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        # choose sensible numeric columns for neighbors
        knn_cols = [c for c in ["median_income", "housing_median_age", "total_rooms",
                                "total_bedrooms", "population", "households",
                                "latitude", "longitude"] if c in numeric_cols]

        if len(knn_cols) >= 2 and len(df) > 1:
            # 1) Impute missing numeric values in the dataset (median imputation)
            imputer = SimpleImputer(strategy="median")
            X_knn = df[knn_cols].copy()
            X_knn_imputed = pd.DataFrame(
                imputer.fit_transform(X_knn),
                columns=knn_cols,
                index=X_knn.index
            )

            # 2) Prepare the query row (user input) and impute any missing values the same way
            query_array = X_example[knn_cols].iloc[0].to_numpy().reshape(1, -1)
            query_imputed = imputer.transform(query_array)

            # 3) Fit NearestNeighbors on the imputed numeric data
            try:
                n_neighbors = min(6, len(df))  # cannot ask more neighbors than samples
                nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_knn_imputed)
                distances, idxs = nbrs.kneighbors(query_imputed)

                # Exclude the query itself if it matches a dataset row (take neighbors 1..)
                # idxs is shape (1, n_neighbors)
                result_idxs = idxs[0]
                # If first neighbor is the same row, drop it (safe guard)
                if len(result_idxs) > 1 and result_idxs[0] == X_knn_imputed.index[0]:
                    result_idxs = result_idxs[1:]
                similar = df.iloc[result_idxs[:5]]  # show up to 5 similar rows

                st.markdown("**Nearby / similar houses from dataset**")
                # show only target + knn numeric columns for readability
                cols_to_show = [TARGET] + knn_cols
                st.dataframe(similar[cols_to_show].reset_index(drop=True))
            except Exception as e:
                st.warning(f"Could not compute similar houses: {e}")
        else:
            st.info("Not enough numeric columns or samples to find similar houses.")

