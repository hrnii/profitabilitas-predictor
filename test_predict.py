import pandas as pd
import joblib

def add_menuitem_freq(X_df):
    # Pastikan fungsi ini identik dengan yang dipakai saat training
    import joblib
    import pandas as pd

    # Load kamus frekuensi dari file yang juga kamu simpan
    menu_item_freq = joblib.load('menu_item_freq.pkl')

    X_df = X_df.copy()
    X_df['MenuItem_freq'] = X_df['MenuItem'].map(menu_item_freq).fillna(0)
    return X_df[['MenuItem_freq']]

# --- 1. Load pipeline dan model
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model = joblib.load('xgb_model.pkl')

# --- 2. Buat data baru (contoh)
data_baru = pd.DataFrame([{
    'RestaurantID': 'R003',
    'MenuCategory': 'Desserts',
    'Ingredients': 'Chocolate Butter Sugar Eggs',
    'MenuItem': 'Newyork Cheesecake',
    'Price': 18.66
}])

# --- 3. Preprocessing
data_baru_preprocessed = preprocessor.transform(data_baru)

# --- 4. Prediksi
pred_encoded = model.predict(data_baru_preprocessed)
pred_label = label_encoder.inverse_transform(pred_encoded)

# --- 5. Output hasil
print(f"Prediksi Profitabilitas: {pred_label[0]}")