import streamlit as st
import pickle
import numpy as np

# BACKGROUND COLOR

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #fdfbfb,
        #ebedee,
        #d7e1ec,
        #f5f7fa
    );
    color: black;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL

model = pickle.load(open("model.pkl", "rb"))

# LOAD SCALER

scaler = pickle.load(open("scaler.pkl", "rb"))

# TITLE

st.title("Online Shopper Purchase Prediction")

st.write("Enter Customer Browsing Details")

# USER INPUTS
#Number of administrative pages visited. Higher values may indicate a more engaged user, while lower values may suggest a casual browser.

administrative = st.number_input(
    "Administrative Pages",
    min_value=0,
    max_value=30,
    value=2
)
#Total time spent on administrative pages. Longer durations may indicate a more engaged user, while shorter durations may suggest a casual browser.
administrative_duration = st.number_input(
    "Administrative Duration",
    min_value=0.0,
    max_value=5000.0,
    value=50.0
)
#Number of informational pages visited. Higher values may indicate a more engaged user, while lower values may suggest a casual browser.
#Users researching products often visit these pages. 
informational = st.number_input(
    "Informational Pages",
    min_value=0,
    max_value=20,
    value=1
)
#Time spent on informational pages .Customer is researching carefully.
informational_duration = st.number_input(
    "Informational Duration",
    min_value=0.0,
    max_value=5000.0,
    value=20.0
)
'''Number of product pages viewed.

Examples:

Mobile page
Laptop page
Shoes page
More product page visits usually mean:
Strong interest
Product comparison
Higher chance of purchase'''
product_related = st.number_input(
    "Product Related Pages",
    min_value=0,
    max_value=1000,
    value=20
)

#Total time spent on product pages.
'''Large duration usually means:
Customer is seriously exploring products
Very important for prediction'''
product_related_duration = st.number_input(
    "Product Related Duration",
    min_value=0.0,
    max_value=50000.0,
    value=500.0
)
#Percentage of users leaving website quickly after viewing one page.Low Bounce Rate Good sign. High Bounce Rate May indicate:
bounce_rates = st.number_input(
    "Bounce Rates",
    min_value=0.0,
    max_value=1.0,
    value=0.02,
    format="%.5f"
)
#Percentage of users exiting from a page. High Exit Rates may indicate:
exit_rates = st.number_input(
    "Exit Rates",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    format="%.5f"
)
'''Represents importance/value of pages before purchase.
Higher page value means:
User visited valuable pages
Strong buying intent'''
page_values = st.number_input(
    "Page Values",
    min_value=0.0,
    max_value=500.0,
    value=10.0
)
'''How close visit is to a special shopping day.
Examples:
Diwali
Christmas
Black Friday
New Year Sale'''
special_day = st.slider(
    "Special Day",
    0.0,
    1.0,
    0.0
)

# PREDICT BUTTON

if st.button("Predict Purchase"):

    features = np.array([
        [
            administrative,
            administrative_duration,
            informational,
            informational_duration,
            product_related,
            product_related_duration,
            bounce_rates,
            exit_rates,
            page_values,
            special_day
        ]
    ])

    # SCALE FEATURES

    scaled_features = scaler.transform(features)

    # PREDICT

    prediction = model.predict(scaled_features)

    # DISPLAY RESULT

    if prediction[0] == 1:

        st.success(
            "Customer Will Purchase Product"
        )

    else:

        st.error(
            "Customer Will Not Purchase Product"
        )