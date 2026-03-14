# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT 
# FRONTEND USER STREAMLIT APP FOR API - NO SECURITY
# ----

# To run app (put this in Terminal):
#   streamlit run 09_streamlit/01_app_streamlit.py
#%%
import streamlit as st
import requests
import pandas as pd
import sys
import pathlib
import os

#%%
# NEEDED FOR EMAIL LEAD SCORING TO BE DETECTED
# APPEND PROJECT DIRECTORY TO PYTHONPATH
working_dir = str(pathlib.Path().absolute())
print("working dir=============" ,working_dir)
syst = sys.path.append(working_dir)

import email_lead_scoring as els

ENDPOINT = 'http://localhost:8000'

#%%
# 1.0 TITLE
st.title("Email Lead Scoring Streamlit Frontend App")

# Define the path to the default file
total_path = '09_streamlit/data/leads.csv'
resolved_total_path = os.path.abspath(total_path)


# Define a radio button for file selection
file_option = st.radio(
    "Select File Source",
    ('Use default leads.csv', 'Upload a file')
)

# File uploader for user-uploaded file
uploaded_file = None
if file_option == 'Upload a file':
    uploaded_file = st.file_uploader(
        "Upload Email Subscribers File",
        type=['csv'],
        accept_multiple_files=False
    )

# Load data based on selected option
@st.cache_data()
def load_data(file_path_or_buffer):
    leads_df = pd.read_csv(file_path_or_buffer)
    return leads_df

if file_option == 'Use default leads.csv':
    leads_df = load_data(resolved_total_path)
    st.write("Using the default leads.csv file")
    full_data_json = leads_df.to_json()

    if st.checkbox("Show raw data", key="default_data"):
        st.subheader("Sample of Raw Data (First 10 Rows)")
        st.write(leads_df.head(10))

    st.write("---")
    st.markdown("# Lead Scoring Analysis")
    
    # User Inputs - Add Sliders / Buttons
    estimated_monthly_sales = st.number_input("How much in email sales per month ($ on average)", 0, value = 250000, step = 1000)
    
    monthly_sales_reduction_safe_guard = st.slider(
        "How much of the monthly sales should be maintained (%)?",
        0., 1., 0.9, step = 0.01
    )
    print(monthly_sales_reduction_safe_guard)
    
    sales_limit = "${:,.0f}".format(monthly_sales_reduction_safe_guard*estimated_monthly_sales)
    
    st.subheader(f"Month sales will not go below: {sales_limit}")

    if st.button("Run Analysis"):
    
        # Spinner
        with st.spinner("Lead scoring in progress. Almost done..."):
        
            # Make Request
            res = requests.post(
                url = f'{ENDPOINT}/calculate_lead_strategy',
                json = full_data_json,
                params = dict(
                    monthly_sales_reduction_safe_guard=float(monthly_sales_reduction_safe_guard),
                    email_list_size=100000,
                    unsub_rate_per_sales_email=0.005,
                    sales_emails_per_month=5,
                    avg_sales_per_month=float(estimated_monthly_sales),
                    avg_sales_emails_per_month=5,
                    customer_conversion_rate=0.05,
                    avg_customer_value=2000.0,
                )
            )
            
            # Collect JSON / Convert Data\
            st.write(f"Response Status Code: {res.status_code}")
            st.write(f"Response Text: {res.text}")

            if res.status_code == 200:
                try:
                    data = res.json()
                    st.write(f"Response JSON Keys: {data.keys()}")
                except ValueError:
                    st.write("Response is not in JSON format.")
            else:
                st.write("Request failed. Check server logs for details.")

            
            lead_strategy_df = pd.read_json(res.json()['lead_strategy'])
            
            expected_value_df = pd.read_json(res.json()['expected_value'])
            
            thresh_optim_table_df = pd.read_json(res.json()['thresh_optim_table'])
            
            print(thresh_optim_table_df.head(10))
            print(expected_value_df.head(10))
            
            
            # Display Results
            st.success("Success! Lead Scoring is complete. Download the results below.")
            
            # Display Strategy Summary
            st.subheader("Lead Strategy Summary:")
            st.write(expected_value_df)
            
            
            # Display Expected Value Plot
            st.subheader("Expected Value Plot")
            st.plotly_chart(
                els.lead_plot_optim_thresh(
                    thresh_optim_table_df,
                    monthly_sales_reduction_safeguard=monthly_sales_reduction_safe_guard
                )
            )
            
            
            # Display Sample Lead Strategy
            st.subheader("Sample of Lead Strategy (First 10 Rows)")
            st.write(lead_strategy_df.head(10))
            
            
            # Download button - Get lead scoring results
            st.download_button(
                label = "Download Lead Scoring Strategy",
                data = lead_strategy_df.to_csv(index = False),
                file_name='lead_strategy.csv',
                mime="text/csv",
                key = "download-csv"
            )
else:
    if uploaded_file is not None:
        leads_df = load_data(uploaded_file)
        st.write("Using the uploaded file")
        if st.checkbox("Show raw data", key="uploaded_data"):
            st.subheader("Sample of Raw Data (First 10 Rows)")
            st.write(leads_df.head(10))
    else:
        st.warning("Please upload a file.")




#%%
# 3.0 APP BODY
if uploaded_file:

    leads_df = load_data(uploaded_file)
    full_data_json = leads_df.to_json()
    
    # Checkbox - Show Table 
    # if st.checkbox("Show raw data"):
    #     st.subheader("Sample of Raw Data (Fisrt 10 Rows)")
    #     st.write(leads_df.head(10))

    st.write("---")
    st.markdown("# Lead Scoring Analysis")
    
    # User Inputs - Add Sliders / Buttons
    estimated_monthly_sales = st.number_input("How much in email sales per month ($ on average)", 0, value = 250000, step = 1000)
    
    monthly_sales_reduction_safe_guard = st.slider(
        "How much of the monthly sales should be maintained (%)?",
        0., 1., 0.9, step = 0.01
    )
    print(monthly_sales_reduction_safe_guard)
    
    sales_limit = "${:,.0f}".format(monthly_sales_reduction_safe_guard*estimated_monthly_sales)
    
    st.subheader(f"Month sales will not go below: {sales_limit}")
    
        # Run Analysis 
    
    if st.button("Run Analysis"):
    
        # Spinner
        with st.spinner("Lead scoring in progress. Almost done..."):
        
            # Make Request
            res = requests.post(
                url = f'{ENDPOINT}/calculate_lead_strategy',
                json = full_data_json,
                params = dict(
                    monthly_sales_reduction_safe_guard=float(monthly_sales_reduction_safe_guard),
                    email_list_size=100000,
                    unsub_rate_per_sales_email=0.005,
                    sales_emails_per_month=5,
                    avg_sales_per_month=float(estimated_monthly_sales),
                    avg_sales_emails_per_month=5,
                    customer_conversion_rate=0.05,
                    avg_customer_value=2000.0,
                )
            )
            
            # Collect JSON / Convert Data\
            
            print(res.json().keys())
            
            lead_strategy_df = pd.read_json(res.json()['lead_strategy'])
            
            expected_value_df = pd.read_json(res.json()['expected_value'])
            
            thresh_optim_table_df = pd.read_json(res.json()['thresh_optim_table'])
            
            print(thresh_optim_table_df.head(10))
            print(expected_value_df.head(10))
            
            
            # Display Results
            st.success("Success! Lead Scoring is complete. Download the results below.")
            
            # Display Strategy Summary
            st.subheader("Lead Strategy Summary:")
            st.write(expected_value_df)
            
            
            # Display Expected Value Plot
            st.subheader("Expected Value Plot")
            st.plotly_chart(
                els.lead_plot_optim_thresh(
                    thresh_optim_table_df,
                    monthly_sales_reduction_safeguard=monthly_sales_reduction_safe_guard
                )
            )
            
            
            # Display Sample Lead Strategy
            st.subheader("Sample of Lead Strategy (First 10 Rows)")
            st.write(lead_strategy_df.head(10))
            
            
            # Download button - Get lead scoring results
            st.download_button(
                label = "Download Lead Scoring Strategy",
                data = lead_strategy_df.to_csv(index = False),
                file_name='lead_strategy.csv',
                mime="text/csv",
                key = "download-csv"
            )
        
    

    
    
# %%
