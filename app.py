import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sys
from pathlib import Path
sys.path.append(os.path.abspath('.'))
from src.model_pipeline.predict_pipeline import ModelPredictor

st.set_page_config(page_title="♻️ Waste Prediction", page_icon="♻️", layout="centered")

@st.cache_resource
def load_model():
    # Determine which model to use - focused model is preferred if available
    model_path = "models/waste_mgmt_focused.joblib" if os.path.exists("models/waste_mgmt_focused.joblib") else "models/waste_mgmt.joblib"
    st.session_state["using_focused_model"] = "focused" in model_path
    
    if st.session_state["using_focused_model"]:
        return ModelPredictor(model_path), None  # No external preprocessor needed for focused model
    else:
        return ModelPredictor(model_path), joblib.load("artifacts/preprocessor.pkl")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("artifacts/train.csv")
    except:
        st.error("Sample data not found. Run training pipeline first.")
        return pd.DataFrame({"city/district": ["Mumbai", "Delhi", "Bengaluru", "Chennai"]})

st.title("♻️ Waste & Recycling Predictor")
st.subheader("Predict using 3 key features")

model, preproc = load_model()
data = load_data()
cities = sorted(data["city/district"].unique()) if "city/district" in data else ["Mumbai", "Delhi", "Bengaluru", "Chennai"]

city = st.selectbox("City/District", cities)
green = st.slider("Green Tech Adoption", 0.0, 100.0, 45.0, 5.0)
infra = st.slider("Recycling Infra Rating", 0.0, 100.0, 55.0, 5.0)

if st.button("Predict Recycling Rate"):
    with st.spinner("Predicting..."):
        # Apply scaling to inputs to better match training data range
        scaled_green = green / 10.0 if green > 10 else green
        
        input_df = pd.DataFrame([{
            "city/district": city,
            "green_technology_adoption": scaled_green,  # Scale down if needed
            "recycling_infrastructure_rating": infra
        }])
        
        try:
            # Handle prediction based on model type
            if st.session_state.get("using_focused_model", False):
                # Focused model has its own preprocessing
                raw_pred = model.predict(input_df)[0]
            else:
                # Use external preprocessor for the standard model
                transformed_features = preproc.transform(input_df)
                raw_pred = model.predict(transformed_features)[0]
            
            # Apply sigmoid-like scaling to naturally limit to 0-100 range
            # This creates a smooth curve that approaches but never exceeds 100
            if raw_pred > 80:
                pred = 80 + 20 * (1 - np.exp(-(raw_pred - 80) / 20))
            else:
                pred = raw_pred
                
            # Cap for safety (should rarely be needed with the scaling above)
            pred = min(max(pred, 0), 100.0)
            
            st.success(f"### Predicted Rate: {pred:.2f}%")
            
            if pred < 30:
                st.warning("Low rate. Improve segregation & awareness.")
            elif pred < 60:
                st.info("Moderate rate. Boost infrastructure.")
            else:
                st.success("Good rate! Keep it up.")
                
            # Added explanation with feature importance for transparency
            with st.expander("See prediction details"):
                st.write("Input factors:")
                st.write(f"- City/District: {city}")
                st.write(f"- Green Tech Adoption: {green}%")
                st.write(f"- Recycling Infrastructure: {infra}%")
                st.write("---")
                st.write("Based on our model, recycling infrastructure and green technology adoption are the strongest predictors of recycling rates.")
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

st.markdown("---")
st.caption("Indian Cities Waste & Recycling - Mini Hackathon")