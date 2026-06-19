import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

MAE = 34.62
LAST_30Days_REVENUE = 6464.50
FORECASTED_REVENUE = 6371.79

# Item Contribution Data
item_data = {
    "Item": ["Sandwich", "Smoothie", "Salad", "Juice", "Cake", "Coffee", "Tea", "Cookie"],
    # Contribution PCT From EDA
    "Contribution": [0.1838, 0.1796, 0.1501, 0.1422, 0.1388, 0.0961, 0.0657, 0.0437],
    "Price": [4.0, 4.0, 5.0, 3.0, 3.0, 2.0, 1.5, 1.0]
}
df_items = pd.DataFrame(item_data)


# APP LAYOUT
st.set_page_config(page_title="Cafe Restock Predictor", layout="wide")

st.title("Cafe Smart Inventory & Restock Predictor")
st.markdown("""
This tool uses a **Random Forest Regressor** to project revenue and recommends 
restocking levels based on the 2023 cafe sales patterns.
""")

# SIDEBAR INPUTS
st.sidebar.header("Control Panel")
revenue_target = st.sidebar.number_input(
    "Set 30 Days Revenue Target ($)", 
    min_value=1000, max_value=10000, value=int(FORECASTED_REVENUE)
)

include_mae = st.sidebar.checkbox("Include MAE Buffer (Safety Stock)", value=True)
promo_active = st.sidebar.checkbox("Activate 'Morning Booster' Bundling (A/B Test Logic)")


# CALCULATION LOGIC
buffer = (MAE * 30) if include_mae else 0
total_needed = revenue_target + buffer

base_atv = 8.47 # From EDA (mean of Total Spent)

# Calculate restocking needs
df_items['Projected_Sales_Val'] = df_items['Contribution'] * total_needed

# Apply Morning Booster Logic
if promo_active:
    # Boost Coffee and Sandwich sales by 10%
    df_items.loc[df_items['Item'].isin(['Coffee', 'Sandwich']), 'Projected_Sales_Val'] *= 1.10
    revenue_lift_factor = (0.0961 + 0.1838) * 0.10 
    projected_atv = base_atv * (1 + revenue_lift_factor)
    atv_delta = projected_atv - base_atv
else:
    projected_atv = base_atv
    atv_delta = 0

df_items['Units_to_Stock'] = (round(df_items['Projected_Sales_Val'] / df_items['Price'])).astype(int)

# Priority
def get_priority(pct):
    if pct > 0.15: return "HIGH (Restock More)"
    if pct > 0.05: return "MEDIUM (Maintain)"
    return "LOW (Reduce Stock)"

df_items['Priority'] = df_items['Contribution'].apply(get_priority)


# DASHBOARD VISUALS
# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Forecast", f"${FORECASTED_REVENUE:,.2f}", "-1.43%")
with col2:
    st.metric(
        label="Safe Stocking Target", 
        value=f"${total_needed:,.2f}",
        help="This is the Model Forecast + Safety Buffer (MAE) to prevent stockouts."
    )
with col2:
    st.metric("Actual Revenue of 30 Days", f"${LAST_30Days_REVENUE:,.2f}")
with col3:
    # KPI ATV INTERAKTIF
    st.metric(
        label="Projected ATV", 
        value=f"${projected_atv:.2f}", 
        delta=f"{atv_delta:.2f}" if promo_active else None,
        help="Average Transaction Value",
        delta_color="normal"
    )

with col4:
    st.metric("Model Reliability (MAE)", f"±${MAE}", "Reasonable (18% MAPE)", delta_color="off")

# Charts & Tables
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Inventory Management Guide")
    
    def style_priority(val):
        color = 'green' if 'HIGH' in val else ('orange' if 'MEDIUM' in val else 'red')
        return f'color: {color}; font-weight: bold'

    st.dataframe(df_items[['Item', 'Priority', 'Units_to_Stock', 'Projected_Sales_Val']]
                 .sort_values('Projected_Sales_Val', ascending=False)
                 .style.map(style_priority, subset=['Priority']))

with col_right:
    st.subheader("Revenue Contribution")
    fig = px.pie(df_items, values='Projected_Sales_Val', names='Item', hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    labels={'Projected_Sales_Val': 'Projected Revenue ($)'},
                    
                 )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        texttemplate='%{label}<br>%{percent:.2%}'
    )
    st.plotly_chart(fig, width="content")


with col_left:
    # Grouped Bar Chart
    st.subheader("Forecast vs. Safe Procurement Target")
    
    df_compare = df_items[['Item', 'Projected_Sales_Val']].copy()
    df_compare['Forecast_Only'] = df_compare['Projected_Sales_Val'] * (revenue_target / total_needed)
    df_compare = df_compare.rename(columns={'Projected_Sales_Val': 'With_Safety_Buffer'})

    fig_delta = px.bar(df_compare, x='Item', y=['Forecast_Only', 'With_Safety_Buffer'], 
                    barmode='group')
    st.plotly_chart(fig_delta)

# INSIGHTS
st.divider()
st.subheader("Actionable Business Intelligence")

base_col, risk_col, data_col = st.columns(3)

with base_col:
    st.markdown("### Revenue & Inventory")
    if revenue_target < LAST_30Days_REVENUE:
        st.warning(f"Target is below historical levels. Recommendation: Close down excess **{df_items.iloc[-1]['Item']}** stock to free up cash.")
    else:
        st.success("Growth from target detected. Prioritize supply chain stability for high-margin items (Sandwich/Smoothie).")

with risk_col:
    st.markdown("### Risk Management")
    st.error(f"Daily Volatility Buffer: **${MAE}**")
    st.caption(f"Based on our model's MAE, we recommend keeping a cash reserve of **${MAE * 7:.2f}** weekly to cover unexpected sales fluctuations.")

with data_col:
    st.markdown("### Data Governance")
    st.info("System Audit Required")
    st.caption("39.72% of historical transactions had missing 'Location' data. Recommendation: Train staff on POS accuracy to improve next month's prediction")

# Expanded Prescriptive Strategy
with st.expander("See Detailed Prescriptive Methods", expanded=True):
    atv_text = f":green[+${atv_delta:.2f}]" if promo_active else "$0.00"
    
    st.write(f"""
    1.  **Inventory Strategy:** Direct 45% of total procurement budget toward **Sandwich** and **Smoothie** ingredients. These represent the "Business Drivers".
    2.  **Experimental Strategy (A/B Test):** 
        *   {':green[**ACTIVE:**]' if promo_active else ':red[**INACTIVE:**]'} Implementation of the **Morning Booster** bundling.
        *   Target: Conversion of 'Coffee-only' customers to 'Bundle' customers. 
        *   Expected ATV Lift: {atv_text} per transaction.
    3.  **Cost Efficiency:** **Cookie** production should be limited to 5% of total inventory to avoid the "Dead Stock" trap identified in our subgroup analysis.
    """)