import streamlit as st
import pandas as pd

model_weights = pd.read_csv("ridge_weights_skl.csv")
mean_std_list = pd.read_csv("x_means_stds")
intercept = 1.087586
price_ratio = 0
sale_price = 0

w_month = model_weights["SaleMonth"]
w_sqr_ftg = model_weights["TotalFinishedArea"]

month_stats = mean_std_list["SaleMonth"]
sqr_ftg_stats = mean_std_list["TotalFinishedArea"]


st.title("Model Demonstration")
st.set_page_config(page_title="Model Demonstration")

date = st.date_input("What is the current date?")
sqr_ftg = st.number_input("Total Square Footage (ft):", min_value=0.0, format="%.2f")

month = (date.month - month_stats[0])/month_stats[1]
sqr_ftg = (sqr_ftg - sqr_ftg_stats[0])/sqr_ftg_stats[1]


price_ratio = intercept + (w_month * month) + (w_sqr_ftg * sqr_ftg)
price_ratio = price_ratio.iloc[0]
price_ratio = round(price_ratio, 2)
st.write("Estimated Price Ratio:", price_ratio)

appraised_val = st.number_input("Total Appraised Value ($):", min_value=0.0, format="%.2f")
sale_price = price_ratio * appraised_val
st.write("Estimated Sale Price", sale_price)

