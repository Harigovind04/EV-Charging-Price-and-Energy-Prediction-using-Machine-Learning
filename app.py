import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(page_title="EV Charging Predictor", page_icon="⚡", layout="centered")

cover_image = Image.open("cover.jpg")   # or "cover.png"
st.image(cover_image, width='stretch')



st.title("⚡ EV Charging Demand & Cost Predictor")
st.write("Enter session details to predict **Demand (kWh)** and **Total Cost (USD)** ")


# ========= LOAD MODELS =========
@st.cache_resource
def load_models():
    demand_model = joblib.load("lightgbm_demand_model.pkl")
    cost_model   = joblib.load("lightgbm_cost_model.pkl")
    return demand_model, cost_model

lgb_demand, lgb_cost = load_models()

price_map = {
    'hotel':0.34,'resort':0.42,'accommodation':0.29,'apartment':0.15,'company':0.00,
    'public institution':0.35,'golf':0.45,'restaurant':0.39,'market':0.42,
    'sightseeing':0.52,'public parking lot':0.48,'public area':0.45,
    'bus garage':0.38,'camping':0.40
}

cat_features = ['UserID', 'Location', 'ChargerType',
                'hour_location', 'user_location', 'charger_user_combo']


hist_df = pd.read_csv("ChargingRecords.csv")
hist_df["StartDatetime"] = pd.to_datetime(
    hist_df["StartDatetime"], format="%d-%m-%Y %H:%M", errors="coerce"
)
hist_df["EndDatetime"] = pd.to_datetime(
    hist_df["EndDatetime"], format="%d-%m-%Y %H:%M", errors="coerce"
)

hist_df = hist_df.dropna(subset=["StartDatetime", "EndDatetime"])
hist_df["Duration_min"] = (hist_df["EndDatetime"] - hist_df["StartDatetime"]).dt.total_seconds() / 60
hist_df = hist_df[hist_df["Duration_min"] > 0]
hist_df = hist_df[hist_df["Demand"] >= 0.01]

hist_df = hist_df.sort_values(["UserID", "StartDatetime"]).reset_index(drop=True)
GLOBAL_DEMAND_MED = hist_df["Demand"].median()

def compute_hist_and_freq(user_id, location, now_dt):
    # filter past sessions of this user
    user_hist = hist_df[
        (hist_df["UserID"].astype(str) == str(user_id)) &
        (hist_df["StartDatetime"] < now_dt)
    ].sort_values("StartDatetime")

    demand_lag1 = GLOBAL_DEMAND_MED
    demand_lag3 = GLOBAL_DEMAND_MED
    demand_mean_5 = GLOBAL_DEMAND_MED
    user_freq = 0

    if len(user_hist) > 0:
        user_freq = len(user_hist)
        demand_lag1 = user_hist["Demand"].iloc[-1]
        demand_mean_5 = user_hist["Demand"].tail(5).mean()
        demand_lag3 = user_hist["Demand"].iloc[-3] if len(user_hist) >= 3 else demand_lag1

    # location frequency 
    loc_hist = hist_df[
        (hist_df["Location"] == location) &
        (hist_df["StartDatetime"] < now_dt)
    ]
    location_freq = len(loc_hist)

    return float(demand_lag1), float(demand_lag3), float(demand_mean_5), int(user_freq), int(location_freq)


# ========= USER INPUTS ONLY (VISIBLE) =========
st.subheader("Session Inputs")

col1, col2 = st.columns(2)
with col1:
    user_id = st.text_input("User ID", "123")
    location = st.selectbox("Location", list(price_map.keys()))
    charger_type = st.radio("Charger Type", [0, 1],
                            format_func=lambda x: "0 - Slow" if x == 0 else "1 - Fast")
with col2:
    day_of_week = st.selectbox("Day of Week (0=Mon .. 6=Sun)", list(range(7)), index=3)
    start_hour = st.number_input("Start Hour (0-23)", min_value=0, max_value=23, value=18)
    start_minute = st.number_input("Start Minute (0-59)", min_value=0, max_value=59, value=30)
    duration_min = st.number_input("Duration (minutes)", min_value=1.0, max_value=600.0, value=60.0)

month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=12)

# ========= DERIVED FEATURES =========
is_weekend = int(day_of_week >= 5)
quarter = (month - 1) // 3 + 1
is_peak_hour = int(start_hour in [7, 8, 17, 18, 19])

hour_location = f"{start_hour}_{location}"
user_location = f"{user_id}_{location}"
charger_user_combo = f"{user_id}_{charger_type}"

# ========= HIDDEN AUTO FEATURES (HARDCODED DEFAULTS) =========

start_dt_str = f"01-{month:02d}-2025 {start_hour:02d}:{start_minute:02d}" 
now_dt = pd.to_datetime(start_dt_str, format="%d-%m-%Y %H:%M")

    # ==== AUTO history & frequency ====
demand_lag1, demand_lag3, demand_mean_5, user_freq, location_freq = \
        compute_hist_and_freq(user_id, location, now_dt)

hour_location = f"{start_hour}_{location}"
user_location = f"{user_id}_{location}"
charger_user_combo = f"{user_id}_{charger_type}"	

# ========= BUILD FEATURE ROW =========
row = {
    'UserID': [str(user_id)],
    'Location': [location],
    'ChargerType': [str(charger_type)],
    'hour_location': [hour_location],
    'user_location': [user_location],
    'start_hour': [start_hour],
    'start_minute': [start_minute],
    'day_of_week': [day_of_week],
    'is_weekend': [is_weekend],
    'month': [month],
    'quarter': [quarter],
    'is_peak_hour': [is_peak_hour],
    'demand_lag1': [demand_lag1],
    'demand_lag3': [demand_lag3],
    'demand_mean_5': [demand_mean_5],
    'Duration_min': [duration_min],
    'user_freq': [user_freq],
    'location_freq': [location_freq],
    'charger_user_combo': [charger_user_combo],
}

input_df = pd.DataFrame(row)

for col in cat_features:
    input_df[col] = input_df[col].astype("category")

# ========= PREDICT =========
if st.button("Predict Demand & Cost"):
    demand_pred = float(lgb_demand.predict(input_df)[0])
    cost_pred   = float(lgb_cost.predict(input_df)[0])

    

    st.success(f"Predicted Demand: **{demand_pred:.2f} kWh**")
    st.success(f"Predicted Cost: **${cost_pred:.2f}**")
   

    

st.markdown("---")
st.caption("Powered by LightGBM with optimized hyperparameters and engineered EV charging features.")
