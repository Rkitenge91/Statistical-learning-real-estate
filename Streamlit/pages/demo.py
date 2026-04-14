import streamlit as st
import pandas as pd

model_weights = pd.read_csv("ridge_weights_skl.csv")

intercept = model_weights["intercept"]
w_month = model_weights["SaleMonth"]
w_year = model_weights["SaleYear"]
w_sqr_ftg = model_weights["TotalFinishedArea"]

st.title("Model Demonstration")
st.set_page_config(page_title="Data Demo")

date = st.date_input("What is the current date?")
month = date.month
year = date.year
sqr_ftg = st.number_input("Total Square Footage (ft):", min_value=0.0, format="%.2f")

price_ratio = intercept + (w_month * month) + (w_year * year) + (w_sqr_ftg * sqr_ftg)

st.write("Price Ratio:", price_ratio)

appraised_val = st.number_input("Total Appraised Value ($):", min_value=0.0, format="%.2f")
sale_price = price_ratio * appraised_val

st.write("Estimated Sale Price:", sale_price)