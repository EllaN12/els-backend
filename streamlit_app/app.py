#%%
import os
import streamlit as st
import requests
import pandas as pd
from helpers import lead_plot_optim_thresh
from constants import ENDPOINT

# Resolve paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")



st.set_page_config(page_title="Email Lead Scoring", page_icon="🚀")
st.title("Email Lead Scoring")


file_option = st.radio(
    "Select File Source",
    ('Use default leads.csv', 'Upload a file')
)

@st.cache_data()
def load_data(file_path_or_buffer):
    return pd.read_csv(file_path_or_buffer)

def run_analysis(full_data_json, monthly_sales_reduction_safe_guard, estimated_monthly_sales):
    with st.spinner("Lead scoring in progress. Almost done..."):
        res = requests.post(
            url=f'{ENDPOINT}/calculate_lead_strategy',
            json=full_data_json,
            params={
                'monthly_sales_reduction_safe_guard': float(monthly_sales_reduction_safe_guard),
                'email_list_size': 100000,
                'unsub_rate_per_sales_email': 0.005,
                'sales_emails_per_month': 5,
                'avg_sales_per_month': float(estimated_monthly_sales),
                'avg_sales_emails_per_month': 5,
                'customer_conversion_rate': 0.05,
                'avg_customer_value': 2000.0,
            }
        )
    return res

def display_results(data, monthly_sales_reduction_safe_guard, key_suffix):
    lead_strategy_df = pd.read_json(data['lead_strategy'])
    expected_value_df = pd.read_json(data['expected_value'])
    thresh_optim_table_df = pd.read_json(data['thresh_optim_table'])

    st.success("Success! Lead Scoring is complete. Download the results below.")
    
    st.subheader("Lead Strategy Summary:")
    st.write(expected_value_df)
    
    st.subheader("Expected Value Plot")
    fig = lead_plot_optim_thresh(
        thresh_optim_table_df,
        monthly_sales_reduction_safeguard=monthly_sales_reduction_safe_guard
    )
    st.plotly_chart(fig)
    
    st.subheader("Sample of Lead Strategy (First 10 Rows)")
    st.write(lead_strategy_df.head(10))
    
    st.download_button(
        label="Download Lead Scoring Strategy",
        data=lead_strategy_df.to_csv(index=False),
        file_name=f'lead_strategy_{key_suffix}.csv',
        mime="text/csv",
        key=f"download-csv-{key_suffix}"
    )

def process_data(leads_df, file_source):
    full_data_json = leads_df.to_json()
    
    if st.checkbox("Show raw data", key=f"{file_source}_data"):
        st.subheader("Sample of Raw Data (First 10 Rows)")
        st.write(leads_df.head(10))

    st.write("---")
    st.markdown("# Lead Scoring Analysis")
    
    estimated_monthly_sales = st.number_input("How much in email sales per month ($ on average)", 0, value=250000, step=1000)
    monthly_sales_reduction_safe_guard = st.slider(
        "How much of the monthly sales should be maintained (%)?",
        0., 1., 0.9, step=0.01
    )
    
    sales_limit = "${:,.0f}".format(monthly_sales_reduction_safe_guard * estimated_monthly_sales)
    st.subheader(f"Month sales will not go below: {sales_limit}")

    if st.button("Run Analysis", key=f"run_analysis_{file_source}"):
        res = run_analysis(full_data_json, monthly_sales_reduction_safe_guard, estimated_monthly_sales)
        
        st.write(f"Response Status Code: {res.status_code}")
        if res.status_code == 200:
            try:
                data = res.json()
                display_results(data, monthly_sales_reduction_safe_guard, file_source)
            except Exception as e:
                st.error(f"An error occurred while processing the data: {str(e)}")
        else:
            st.error("Request failed. Check server logs for details.")
            st.write(f"Response Text: {res.text}")

if file_option == 'Use default leads.csv':
    resolved_total_path = os.path.join(_DATA_DIR, "leads.csv")
    leads_df = load_data(resolved_total_path)
    st.write("Using the default leads.csv file")
    process_data(leads_df, "default")
else:
    uploaded_file = st.file_uploader(
        "Upload Email Subscribers File",
        type=['csv'],
        accept_multiple_files=False
    )
    if uploaded_file is not None:
        leads_df = load_data(uploaded_file)
        st.write("Using the uploaded file")
        process_data(leads_df, "uploaded")
    else:
        st.warning("Please upload a file.")





# %%
