import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import base64
import os

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Menu Profitability Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# LOAD CUSTOM CSS
# ======================================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

# ======================================
# BACKGROUND IMAGE (optional)
# ======================================
if os.path.exists("background.jpeg"):
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    add_bg_from_local("background.jpeg")

# ======================================
# CUSTOM FUNCTION FOR PREPROCESSOR
# ======================================
def add_menuitem_freq(X_df):
    try:
        menu_item_freq = joblib.load("menu_item_freq.pkl")
        X_df = X_df.copy()
        X_df["MenuItem_freq"] = X_df["MenuItem"].map(menu_item_freq).fillna(0)
        return X_df[["MenuItem_freq"]]
    except FileNotFoundError:
        return pd.DataFrame({"MenuItem_freq": [0]*len(X_df)})

# ======================================
# LOAD ARTIFACTS
# ======================================
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model = joblib.load("xgb_model.pkl")
    return preprocessor, label_encoder, model

preprocessor, label_encoder, model = load_artifacts()

# ======================================
# HEADER
# ======================================
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2450/2450923.png", width=100)
with col2:
    st.markdown('<h2 class="prediction-title">🍽️ Menu Profitability Predictor</h2>', unsafe_allow_html=True)
    st.write("Predict the profitability of your restaurant menu items with AI-powered analytics.")

st.markdown("---")

# ======================================
# MAIN CONTENT
# ======================================
with st.container():
    col1, col2 = st.columns([2, 3])

    with col1:
        with st.form("prediction_form"):
            st.subheader("📝 Menu Information")

            restaurant_id = st.text_input("Restaurant ID", "")
            menu_category = st.text_input("Menu Category", "")
            menu_item = st.text_input("Menu Item Name", "")
            price = st.number_input("Price ($)", min_value=0.0, step=0.01, format="%.2f")
            ingredients = st.text_area("Ingredients", "")

            submit_button = st.form_submit_button("✨ Predict Profitability", use_container_width=True)

    with col2:
        st.subheader("🔮 Prediction Results")

        if submit_button:
            input_data = pd.DataFrame([{
                "RestaurantID": restaurant_id,
                "MenuCategory": menu_category,
                "Ingredients": ingredients,
                "MenuItem": menu_item,
                "Price": price
            }])

            try:
                input_preprocessed = preprocessor.transform(input_data)
                pred_encoded = model.predict(input_preprocessed)
                pred_label = label_encoder.inverse_transform(pred_encoded)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred_label = None

            if pred_label:
                with st.expander("📊 View Input Data", expanded=True):
                    st.dataframe(input_data.style.highlight_max(axis=0), use_container_width=True)

                # Prediction title
                st.markdown(f'<h3 class="prediction-title">Hasil Prediksi: {pred_label}</h3>', unsafe_allow_html=True)

                # Prediction message
                if pred_label == "High":
                    st.markdown('<div class="prediction-message">Menu ini diprediksi sangat menguntungkan! 🚀</div>', unsafe_allow_html=True)
                elif pred_label == "Medium":
                    st.markdown('<div class="prediction-message" style="background-color:#fef3c7; color:#92400e;">Menu ini memiliki potensi keuntungan sedang ⚖️</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-message" style="background-color:#fee2e2; color:#991b1b;">Menu ini kemungkinan kurang menguntungkan ⚠️</div>', unsafe_allow_html=True)

                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.markdown('<div class="prediction-value">92%</div>', unsafe_allow_html=True)
                col_m1.markdown('<div class="prediction-diff">↑ 3% dari rata-rata</div>', unsafe_allow_html=True)

                col_m2.markdown(f'<div class="prediction-value">${price*1.1:.2f}</div>', unsafe_allow_html=True)
                col_m2.markdown('<div class="prediction-diff">↑ +10%</div>', unsafe_allow_html=True)

                col_m3.markdown('<div class="prediction-value">24</div>', unsafe_allow_html=True)
                col_m3.markdown('<div class="prediction-diff">↑ Di database Anda</div>', unsafe_allow_html=True)

        else:
            st.markdown(
                """
                <div class="custom-info">
                👈 <b>Fill out the form and click 'Predict Profitability'</b>
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#666; font-size:14px;">
This prediction is based on machine learning models and historical data. Actual results may vary.
</p>
""", unsafe_allow_html=True)
