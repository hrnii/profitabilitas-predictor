import streamlit as st
import pandas as pd
import joblib
import os

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Menu Profitability Predictor",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Menu Profitability Predictor")
st.write("Masukkan data menu Anda untuk memprediksi tingkat profitabilitasnya.")

# ========================
# LOAD MODEL & ARTIFAK
# ========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("xgb_model.pkl")
        preprocessor = joblib.load("pipeline_preprocessing.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, preprocessor, label_encoder
    except FileNotFoundError as e:
        st.error(f"File model/artifact tidak ditemukan: {e}")
        return None, None, None, None

model, preprocessor, label_encoder = load_artifacts()

# ========================
# FORM INPUT
# ========================
with st.form("prediction_form"):
    restaurant_id = st.text_input("Restaurant ID", "")
    menu_category = st.text_input("Menu Category", "")
    menu_item = st.text_input("Menu Item Name", "")
    price = st.number_input("Price ($)", min_value=0.0, step=0.01, format="%.2f")
    ingredients = st.text_area("Ingredients", "")

    submitted = st.form_submit_button("Predict")

# ========================
# PREDICTION
# ========================
if submitted:
    if not all([restaurant_id, menu_category, menu_item, ingredients]):
        st.warning("⚠️ Harap isi semua kolom sebelum prediksi.")
    elif model is None:
        st.error("❌ Model belum tersedia, prediksi tidak dapat dilakukan.")
    else:
        try:
            input_df = pd.DataFrame([{
                "RestaurantID": restaurant_id,
                "MenuCategory": menu_category,
                "Ingredients": ingredients,
                "MenuItem": menu_item,
                "Price": price
            }])

            # Preprocessing
            X_transformed = preprocessor.transform(input_df)

            # Prediction
            pred_encoded = model.predict(X_transformed)
            pred_label = label_encoder.inverse_transform(pred_encoded)[0]

            # Output
            st.subheader("Hasil Prediksi")
            if pred_label == "High":
                st.success("🚀 Menu ini diprediksi sangat menguntungkan! (High)")
            elif pred_label == "Medium":
                st.info("⚖️ Menu ini memiliki potensi keuntungan sedang. (Medium)")
            else:
                st.warning("⚠️ Menu ini kemungkinan kurang menguntungkan. (Low)")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
