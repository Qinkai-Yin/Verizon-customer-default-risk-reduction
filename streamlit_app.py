import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Verizon Customer Default Prediction",
    layout="wide",
)

# =========================
# CSS (clean & modern)
# =========================
def inject_css():
    st.markdown(
        """
<style>
/* 页面上下 padding 缩小 */
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; }

/* 控件之间整体间距缩小 */
div[data-testid="stVerticalBlock"] > div { gap: 0.45rem; }

/* label 更紧凑 */
label { margin-bottom: 0.15rem !important; }

/* 让 tabs 下方空白更少（有时有效） */
div[data-testid="stTabs"] { margin-top: -0.4rem; }

/* metric 更紧凑 */
[data-testid="stMetric"] { padding: 0.25rem 0.5rem; }
</style>
""", unsafe_allow_html=True,
    )

inject_css()

# =========================
# Model columns (MUST match X_train.columns)
# =========================
MODEL_COLUMNS = [
    "price",
    "downpmt",
    "monthdue",
    "monthly_payment",
    "credit_score",
    "is_under_18",
    "remaining_ratio",
    "age_filled",
    "gender_2",
    "pmttype_3",
    "pmttype_4",
    "pmttype_5",
]

PAYMENT_TYPES = [
    "Credit Payment",     # pmttype=1 baseline -> all zeros
    "Store Gift Card",    # pmttype=3
    "Debt Payment",       # pmttype=4
    "Cash Payment",       # pmttype=5
]

# =========================
# Helpers
# =========================
@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback: some models output probability directly
    pred = model.predict(X)
    return np.array(pred, dtype=float)

def pretty_confidence(p: float) -> str:
    if p >= 0.80:
        return "High"
    if p >= 0.60:
        return "Medium"
    return "Low"

def build_model_input(
    price: float,
    downpmt: float,
    monthdue: float,
    monthly_payment: float,
    credit_score: float,
    age: float | None,
    gender: str,
    payment_type: str,
) -> pd.DataFrame:
    """
    Convert human-friendly inputs -> model-ready 12 features.
    Aligned with your X_train.columns.
    """

    # age handling (based on your earlier logic)
    if age is None or (isinstance(age, float) and np.isnan(age)):
        is_under_18 = 1
        age_filled = 17
    else:
        is_under_18 = int(age < 18)
        age_filled = float(age)

    # remaining_ratio (change here if your original formula differs)
    remaining_ratio = (price - downpmt) / price if price != 0 else 0.0

    # gender_2 dummy: Female -> 1, else 0
    gender_2 = 1 if str(gender).strip().lower() in ["female", "f"] else 0

    # payment type dummies based on your confirmed mapping
    pmttype_3 = int(payment_type == "Store Gift Card")
    pmttype_4 = int(payment_type == "Debt Payment")
    pmttype_5 = int(payment_type == "Cash Payment")

    row = {
        "price": float(price),
        "downpmt": float(downpmt),
        "monthdue": float(monthdue),
        "monthly_payment": float(monthly_payment),
        "credit_score": float(credit_score),
        "is_under_18": int(is_under_18),
        "remaining_ratio": float(remaining_ratio),
        "age_filled": float(age_filled),
        "gender_2": int(gender_2),
        "pmttype_3": int(pmttype_3),
        "pmttype_4": int(pmttype_4),
        "pmttype_5": int(pmttype_5),
    }

    return pd.DataFrame([row], columns=MODEL_COLUMNS)

def find_model_path(choice: str) -> str | None:
    # You can add more candidates if needed
    candidates = {
        "XGBoost": ["Verizon_xgb_model.pkl", os.path.join("models", "Verizon_xgb_model.pkl")],
        "LightGBM": ["Verizon_lightgbm_model.pkl", os.path.join("models", "Verizon_lightgbm_model.pkl")],
    }
    for p in candidates.get(choice, []):
        if os.path.exists(p):
            return p

    # auto fallback (first found)
    auto = [
        "Verizon_xgb_model.pkl",
        "Verizon_lightgbm_model.pkl",
        os.path.join("models", "Verizon_xgb_model.pkl"),
        os.path.join("models", "Verizon_lightgbm_model.pkl"),
    ]
    for p in auto:
        if os.path.exists(p):
            return p
    return None

# =========================
# Header
# =========================
h1, h2 = st.columns([0.7, 0.3], vertical_alignment="top")
with h1:
    st.markdown('<div class="title">Verizon — Customer Default Prediction</div>', unsafe_allow_html=True)
 
st.write("")

# =========================
# Sidebar settings
# =========================
with st.sidebar:
    st.markdown("### Settings")
    model_choice = st.selectbox("Model", ["XGBoost", "LightGBM"], index=0)
    model_path = find_model_path(model_choice)
    if model_path is None:
        st.error("Model .pkl not found. Put it in project root or ./models/")
        st.stop()

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    st.caption(f"Using: `{model_path}`")

model = load_model(model_path)

def build_X_from_excel(df: pd.DataFrame) -> pd.DataFrame:
    X_list = []

    for _, r in df.iterrows():
        price = r["price"]
        downpmt = r["downpmt"]
        monthdue = r["monthdue"]
        monthly_payment = r["monthly_pay"]  
        credit_score = r["credit_score"]
        age = r["age"]
        gender = r["gender"]
        pmttype = r["pmttype"]

        # ---- age features ----
        if pd.isna(age):
            is_under_18 = 1
            age_filled = 28
        else:
            is_under_18 = int(age < 18)
            age_filled = age

        # ---- remaining ratio ----
        remaining_ratio = (price - downpmt) / price if price != 0 else 0.0

        # ---- gender dummy (gender_2) ----
        gender_2 = int(gender == 2)

        # ---- pmttype dummies ----
        pmttype_3 = int(pmttype == 3)
        pmttype_4 = int(pmttype == 4)
        pmttype_5 = int(pmttype == 5)

        X_list.append({
            "price": price,
            "downpmt": downpmt,
            "monthdue": monthdue,
            "monthly_payment": monthly_payment,
            "credit_score": credit_score,
            "is_under_18": is_under_18,
            "remaining_ratio": remaining_ratio,
            "age_filled": age_filled,
            "gender_2": gender_2,
            "pmttype_3": pmttype_3,
            "pmttype_4": pmttype_4,
            "pmttype_5": pmttype_5,
        })

    return pd.DataFrame(X_list)


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Scoring", "Notes"])

# =========================
# Tab 1: Single prediction
# =========================
with tab1:
    st.markdown("### Single Prediction")
    st.caption("Estimate default probability and return an approval decision.")

    # -------- Inputs (compact grid) --------
    c1, c2, c3 = st.columns(3)
    with c1:
        price = st.number_input("Price", min_value=0.0, value=1000.0, step=50.0)
    with c2:
        downpmt = st.number_input("Down Payment", min_value=0.0, value=200.0, step=50.0)
    with c3:
        credit_score = st.number_input("Credit Score", min_value=0, max_value=8, value=2, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        monthdue = st.number_input("Month Due", min_value=0, value=8, step=1)
    with c5:
        monthly_payment = st.number_input("Monthly Payment", min_value=0.0, value=150.0, step=10.0)
    with c6:
        payment_type = st.selectbox(
            "Payment Type",
            ["Credit Payment", "Store Gift Card", "Debt Payment", "Cash Payment"],
            index=0
        )

    c7, c8, c9 = st.columns(3)
    with c7:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    with c8:
        age_unknown = st.checkbox("Age unknown", value=False)
    with c9:
        age = np.nan if age_unknown else st.number_input("Age", min_value=0, max_value=120, value=25, step=1)

    # ---- run button (single row, compact) ----
    run = st.button("Run Risk Check", use_container_width=True, key="run_single")

    # ===============================
    # BOTTOM: OUTPUT
    # ===============================
    if run:
        try:
            X_one = build_model_input(
                price=price,
                downpmt=downpmt,
                monthdue=monthdue,
                monthly_payment=monthly_payment,
                credit_score=credit_score,
                age=age,
                gender=gender,
                payment_type=payment_type,
            )

            proba_default = predict_proba(model, X_one)

            default_proba = float(proba_default)

            decision = "Reject" if default_proba >= threshold else "Approve"


    
            col1, col2 = st.columns([1.3, 1], gap="small")

            # ---- Left: Decision ----
            with col1:
                decision_color = "#c62828" if decision == "Reject" else "#2e7d32"

                st.markdown(
                    f"""
                    <div>
                        <div style="font-size:18px; font-weight:400; margin-bottom:6px;">
                            Decision
                        </div>
                        <div style="font-size:40px; font-weight:700; color:{decision_color};">
                            {decision}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ---- Right: Default Probability ----
            with col2:
                st.markdown(
                    f"""
                    <div>
                        <div style="font-size:18px; font-weight:400; margin-bottom:6px;">
                            Default Probability
                        </div>
                        <div style="font-size:40px; font-weight:400;">
                            {default_proba*100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # 默认收起，不占高度
            with st.expander("Show model input (12 features)"):
                st.dataframe(X_one, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Enter inputs above and click **Run Risk Check**.")

# =========================
# Tab 2: Batch scoring
# =========================
import io

with tab2:
    st.markdown("## Batch Scoring")
    st.caption("Upload a CSV or Excel file to run batch default risk scoring.")

    REQUIRED_RAW_COLS = ["price","downpmt","monthdue","monthly_payment","credit_score","age","gender","payment_type"]

    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
        
    def read_uploaded_table(uploaded_file) -> pd.DataFrame:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)  # needs openpyxl
        raise ValueError("Unsupported file type. Please upload .csv or .xlsx")

    def build_X_for_batch(df: pd.DataFrame) -> pd.DataFrame:
        # Case A: Excel already contains engineered model columns (12 features)
        if all(c in df.columns for c in MODEL_COLUMNS):
            return df[MODEL_COLUMNS].copy()

        # Case B: Excel contains human-friendly columns -> map row by row
        missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. "
                            f"Provide either MODEL_COLUMNS={MODEL_COLUMNS} OR raw columns={REQUIRED_RAW_COLS}")

        X_list = []
        for _, r in df.iterrows():
            X_list.append(
                build_model_input(
                    price=r["price"],
                    downpmt=r["downpmt"],
                    monthdue=r["monthdue"],
                    monthly_payment=r["monthly_payment"],
                    credit_score=r["credit_score"],
                    age=r["age"],
                    gender=r["gender"],
                    payment_type=r["payment_type"],
                )
            )
        return pd.concat(X_list, ignore_index=True)
    
    if uploaded is not None:
        df_in = read_uploaded_table(uploaded)
        st.dataframe(df_in.head(30), use_container_width=True)

# --- In your Batch tab ---

    if st.button("Run batch scoring", use_container_width=True):
            try:
                X_model = build_X_for_batch(df_in)
                default_proba = predict_proba(model, X_model)

                out = df_in.copy()
                out["default_probability"] = default_proba
                out["decision"] = np.where(out["default_probability"] >= threshold, "Reject", "Approve")

                st.success("Done.")
                st.dataframe(out.head(50), use_container_width=True)

                # Download CSV
                st.download_button(
                    "Download results (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="default_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # Download Excel
                buf = io.BytesIO()
                out.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "Download results (Excel)",
                    data=buf.getvalue(),
                    file_name="default_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Batch scoring failed: {e}")

# =========================
# Tab 3: Notes / sanity checks
# =========================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Notes")
    st.write(
        """
- This app **does NOT use a pipeline**. It converts human-friendly inputs into your model’s 12 engineered features.
- Payment type mapping is fixed:
  - 1 → Credit Payment (baseline)
  - 3 → Store Gift Card
  - 4 → Debt Payment
  - 5 → Cash Payment
- If your original `remaining_ratio` formula differs, change it in `build_model_input()`.
        """
    )
    st.markdown("**Sanity check:** model feature columns used:")
    st.code(", ".join(MODEL_COLUMNS), language="text")
    st.markdown('</div>', unsafe_allow_html=True)
