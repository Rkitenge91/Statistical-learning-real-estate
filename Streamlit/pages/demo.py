import streamlit as st
import pandas as pd

model_weights = pd.read_csv("ridge_weights_skl.csv")
x_describe = pd.read_csv("x_describe")
intercept = 1.087586

w_month = model_weights["SaleMonth"]
w_year = model_weights["SaleYear"]
w_sqr_ftg = model_weights["TotalFinishedArea"]
w_living_units = model_weights["LivingUnits"]

month_d = x_describe["SaleMonth"]
year_d = x_describe["SaleYear"]
sqrt_ftg_d = x_describe["TotalFinishedArea"]
living_units_d = x_describe["LivingUnits"]


st.title("Model Demonstration")
st.set_page_config(page_title="Model Demonstration")

date = st.date_input("Sale date?")
sqr_ftg = st.number_input("Total Square Footage (ft):", min_value=sqrt_ftg_d.iloc[3], max_value=sqrt_ftg_d.iloc[7], format="%.2f")
living_units = st.number_input("Number of Living Units:", min_value=living_units_d.iloc[3], max_value=living_units_d.iloc[7], format="%.2f")

#if date.year == 2024 or date.year == 2025:
#    year = date.year
#else:
#    year = 0

month = (date.month - month_d.iloc[1])/month_d.iloc[2]
year = (date.year - year_d.iloc[1])/year_d.iloc[2]
sqr_ftg = (sqr_ftg - sqrt_ftg_d.iloc[1])/sqrt_ftg_d.iloc[2]
living_units = (living_units - living_units_d.iloc[1])/living_units_d.iloc[2]


price_ratio = intercept + (w_month * month) + (w_sqr_ftg * sqr_ftg) + (w_living_units * living_units)
price_ratio = price_ratio.iloc[0]
price_ratio = round(price_ratio, 2)
st.write("Estimated Price Ratio:", price_ratio)

appraised_val = st.number_input("Total Appraised Value ($):", min_value=0.0, format="%.2f")
sale_price = price_ratio * appraised_val
st.write("Estimated Sale Price", sale_price)

