import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ======================================================
# CONFIG – EDIT THESE TO MATCH YOUR SOCIOECONOMIC FILE
# ======================================================

# Name/path of your socioeconomic regression CSV (same directory as this app)
SOCIO_FILE = "socioeconomic_regression_data.csv"  # <-- change if needed

# Outcome variable (Y)
DEFAULT_TARGET = "avg_att8"  # Attainment 8

# Preferred socioeconomic predictors (X)
PREFERRED_NUMERIC = [
    "pt_fsm6",       # % FSM6
    "pt_fsm",        # % FSM (if present)
    "idaci_score",   # IDACI continuous score
    "pt_eal",        # % EAL
    "pt_sen_sup",    # % SEN Support
    "pt_ehcp",       # % EHCP
]

PREFERRED_CATEGORICAL = [
    "ofsted_num",      # if numeric rating is stored as int, will be treated as numeric
    "ofsted_rating",   # text rating, if present
    "region_name",
    "school_type",
    "urban_rural",
]


# ======================================================
# STREAMLIT APP SETUP
# ======================================================

st.set_page_config(
    page_title="Socioeconomic Regression Dashboard",
    layout="wide",
)

st.title("📊 Socioeconomic Regression Dashboard (Local Data, No Statsmodels)")

st.markdown(
    """
This app loads a **local socioeconomic regression file**, runs a **linear regression** using
`scikit-learn` (no `statsmodels`), and shows **Plotly visualisations**.

- Outcome: typically **Attainment 8** (`avg_att8`)
- Predictors: FSM, IDACI, EAL, SEN, Ofsted rating, region, school type, etc.
"""
)

# ======================================================
# LOAD SOCIOECONOMIC DATA
# ======================================================

st.sidebar.header("Data Source")

file_path = st.sidebar.text_input(
    "Path to socioeconomic CSV file",
    value=SOCIO_FILE,
    help="This should be the CSV with merged socioeconomic + performance data.",
)

if not os.path.exists(file_path):
    st.error(f"File not found at `{file_path}`. Put the file in the same directory or update the path.")
    st.stop()

try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

st.subheader("Preview of Socioeconomic Data")
st.caption(f"Loaded from: `{file_path}`")
st.dataframe(df.head())

if df.empty:
    st.error("The loaded dataset is empty.")
    st.stop()

# ======================================================
# VARIABLE SELECTION (SOCIOECONOMIC MODEL)
# ======================================================

st.sidebar.header("Model Configuration")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found. You need at least one numeric column for the outcome variable.")
    st.stop()

# Outcome (Y)
if DEFAULT_TARGET in numeric_cols:
    default_target_index = numeric_cols.index(DEFAULT_TARGET)
else:
    default_target_index = 0  # fallback to first numeric column

target_col = st.sidebar.selectbox(
    "Select outcome (dependent variable)",
    numeric_cols,
    index=default_target_index,
)

# Predictors (X): intersect PREFERRED lists with actual columns
default_predictors = []

for col in PREFERRED_NUMERIC + PREFERRED_CATEGORICAL:
    if col in all_cols and col != target_col:
        default_predictors.append(col)

# if intersection is empty, fallback to all non-target columns
if not default_predictors:
    default_predictors = [c for c in all_cols if c != target_col]

predictor_cols = st.sidebar.multiselect(
    "Select predictors (independent variables)",
    options=[c for c in all_cols if c != target_col],
    default=default_predictors,
)

if not predictor_cols:
    st.warning("Please select at least one predictor.")
    st.stop()

# Split predictors into numeric vs categorical for preprocessing
cat_guess = [c for c in predictor_cols if c in non_numeric_cols]
num_guess = [c for c in predictor_cols if c in numeric_cols]

categorical_cols = st.sidebar.multiselect(
    "Treat these predictors as categorical",
    options=predictor_cols,
    default=cat_guess,
)

numerical_cols = [c for c in predictor_cols if c not in categorical_cols]

st.sidebar.markdown("---")
test_size = st.sidebar.slider(
    "Test set size (proportion)",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
)
random_state = st.sidebar.number_input(
    "Random state (for reproducibility)",
    min_value=0,
    max_value=9999,
    value=42,
)

run_button = st.sidebar.button("🚀 Run socioeconomic regression")

if not run_button:
    st.stop()

# ======================================================
# DATA CLEANING
# ======================================================

model_df = df[[target_col] + predictor_cols].dropna()
n_dropped = len(df) - len(model_df)

if model_df.empty:
    st.error("After dropping missing values in Y and X, there is no data left. Check your columns or missingness.")
    st.stop()

st.warning(f"Dropped {n_dropped} rows due to missing values. Using {len(model_df)} rows for modelling.")

X = model_df[predictor_cols].copy()
y = model_df[target_col].copy()

# ======================================================
# PREPROCESSING & PIPELINE (NO STATSMODELS)
# ======================================================

transformers = []

if numerical_cols:
    transformers.append(("num", StandardScaler(), numerical_cols))

if categorical_cols:
    transformers.append(
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

regressor = LinearRegression()

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# ======================================================
# METRICS ON DASHBOARD
# ======================================================

st.subheader("📈 Socioeconomic Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train R²", f"{train_r2:.3f}")
col2.metric("Test R²", f"{test_r2:.3f}")
col3.metric("Test MAE", f"{test_mae:.3f}")
col4.metric("Test RMSE", f"{test_rmse:.3f}")

st.markdown(
    f"""
**Outcome (Y):** `{target_col}`  
**Predictors (X):** {", ".join(predictor_cols)}
"""
)

# ======================================================
# FEATURE NAMES & COEFFICIENTS
# ======================================================

feature_names = []

if numerical_cols:
    feature_names.extend(numerical_cols)

if categorical_cols:
    cat_transformer = model.named_steps["preprocessor"].named_transformers_.get("cat")
    if cat_transformer is not None:
        cat_feature_names = cat_transformer.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names.tolist())

coef = model.named_steps["regressor"].coef_

feature_importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }
).sort_values("abs_coefficient", ascending=False)

# ======================================================
# VISUALISATIONS (PLOTLY) – VISIBLE ON DASHBOARD
# ======================================================

st.subheader("📊 Visualisations")

tab1, tab2, tab3 = st.tabs(
    ["Actual vs Predicted", "Residuals vs Predicted", "Feature Importance"]
)

# 1) Actual vs Predicted (test set)
with tab1:
    st.markdown("### Actual vs Predicted (Test Set)")
    ap_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_test_pred,
        }
    )
    fig_ap = px.scatter(
        ap_df,
        x="Actual",
        y="Predicted",
        labels={"Actual": f"Actual {target_col}", "Predicted": f"Predicted {target_col}"},
        title="Actual vs Predicted (Test Set)",
    )
    # IMPORTANT: no trendline="ols" here (would require statsmodels)
    st.plotly_chart(fig_ap, use_container_width=True)

# 2) Residuals vs Predicted
with tab2:
    st.markdown("### Residuals vs Predicted (Test Set)")
    residuals = y_test - y_test_pred
    res_df = pd.DataFrame(
        {
            "Predicted": y_test_pred,
            "Residuals": residuals,
        }
    )
    fig_res = px.scatter(
        res_df,
        x="Predicted",
        y="Residuals",
        labels={"Predicted": f"Predicted {target_col}", "Residuals": "Residuals"},
        title="Residuals vs Predicted",
    )
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

# 3) Feature Importance
with tab3:
    st.markdown("### Feature Importance (Absolute Coefficients)")
    fig_coef = px.bar(
        feature_importance_df,
        x="abs_coefficient",
        y="feature",
        orientation="h",
        labels={"abs_coefficient": "Absolute coefficient", "feature": "Feature"},
        title="Feature Importance",
    )
    fig_coef.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("Raw coefficients")
    st.dataframe(feature_importance_df[["feature", "coefficient"]])
