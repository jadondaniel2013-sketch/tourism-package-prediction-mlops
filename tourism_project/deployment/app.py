# Import Path so the deployed app can locate the model relative to app.py regardless of working directory.
from pathlib import Path
# Import joblib so the serialized preprocessing-plus-XGBoost pipeline can be loaded.
import joblib
# Import pandas so user inputs can be assembled into the one-row DataFrame expected by the model.
import pandas as pd
# Import Streamlit to build the web user interface served from Streamlit Community Cloud.
import streamlit as st

# Resolve the deployment folder that contains both this app and the trained model artifact.
DEPLOYMENT_DIR = Path(__file__).resolve().parent
# Define the serialized model path committed to GitHub by the MLOps pipeline.
MODEL_PATH = DEPLOYMENT_DIR / "best_tourism_model.joblib"
# Configure the browser tab title and use a wide layout for a compact business form.
st.set_page_config(page_title="Tourism Package Purchase Prediction", page_icon="✈️", layout="wide")
# Load the complete fitted preprocessing-and-classification pipeline once when the app starts.
model = joblib.load(MODEL_PATH)
# Display the app title for the marketing user.
st.title("Tourism Package Purchase Prediction")
# Explain what the prediction represents before the user enters customer information.
st.write("Estimate whether a customer is likely to purchase the Wellness Tourism Package and use the probability to prioritize marketing follow-up.")
# Create two columns so the form remains readable without excessive vertical scrolling.
left_column, right_column = st.columns(2)
# Collect the first half of customer inputs in the left column.
with left_column:
    # Collect customer age as a bounded whole number.
    age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)
    # Collect how the customer entered the sales process.
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    # Collect the customer's city tier as one of the three categories in the dataset.
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    # Collect the pitch duration in minutes.
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=15.0, step=1.0)
    # Collect the standardized occupation category.
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
    # Collect the standardized gender category.
    gender = st.selectbox("Gender", ["Female", "Male"])
    # Collect the number of travelers in the party.
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
    # Collect the number of salesperson follow-ups.
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
    # Collect the tourism package type pitched to the customer.
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
# Collect the remaining customer inputs in the right column.
with right_column:
    # Collect the preferred hotel-star level.
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    # Collect the standardized marital-status category used by the trained model.
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    # Collect the typical number of annual trips.
    number_of_trips = st.number_input("Number of Trips per Year", min_value=0.0, max_value=30.0, value=3.0, step=1.0)
    # Collect whether the customer holds a passport as a binary value.
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    # Collect the pitch satisfaction score on the observed one-to-five scale.
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    # Collect whether the customer owns a car as a binary value.
    own_car = st.selectbox("Owns a Car", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    # Collect the number of young children traveling with the customer.
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=1.0, step=1.0)
    # Collect the customer's organizational designation.
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    # Collect gross monthly income using a range that comfortably covers the training data.
    monthly_income = st.number_input("Monthly Income", min_value=5000.0, max_value=100000.0, value=22000.0, step=500.0)
# Assemble the form values into a one-row DataFrame using exactly the feature names used during training.
input_data = pd.DataFrame([
    {
        "Age": float(age),
        "TypeofContact": type_of_contact,
        "CityTier": int(city_tier),
        "DurationOfPitch": float(duration_of_pitch),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(number_of_person_visiting),
        "NumberOfFollowups": float(number_of_followups),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": float(preferred_property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(number_of_trips),
        "Passport": int(passport),
        "PitchSatisfactionScore": int(pitch_satisfaction_score),
        "OwnCar": int(own_car),
        "NumberOfChildrenVisiting": float(number_of_children_visiting),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income),
    }
])
# Run scoring only when the marketing user explicitly clicks the prediction button.
if st.button("Predict Purchase Likelihood", type="primary"):
    # Predict the positive-class purchase probability so leads can be ranked by propensity.
    purchase_probability = float(model.predict_proba(input_data)[0, 1])
    # Convert the default 0.50 probability threshold into the model's binary decision.
    purchase_prediction = int(purchase_probability >= 0.50)
    # Display the probability prominently because ranking is more useful than only a yes/no label.
    st.metric("Predicted Purchase Probability", f"{purchase_probability:.1%}")
    # Show a concise business action when the model predicts a likely buyer.
    if purchase_prediction == 1:
        # Encourage priority follow-up for customers above the default threshold.
        st.success("Likely buyer: prioritize this customer for timely follow-up.")
    # Show a lower-priority recommendation when the probability is below the default threshold.
    else:
        # Encourage efficient use of sales resources while keeping the probability visible for alternative thresholds.
        st.info("Lower purchase likelihood at the 50% threshold: consider lower-cost nurturing or a later follow-up.")
