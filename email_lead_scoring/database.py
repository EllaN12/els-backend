import pandas as pd
import numpy as np
import sqlalchemy as sql
from sqlalchemy import text
import re
import janitor as jn
from app.email_lead_scoring.exploratory import explore_sales_by_category
import os

# Set up database paths and connection string
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATABASE_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'crm_database.sqlite')
os.makedirs(DATABASE_DIR, exist_ok=True)
CONN_STR = f"sqlite:///{DATABASE_PATH}"

def db_read_els_table_names(conn_str = CONN_STR):
    """
    Read table names from the CRM database.

    Args:
        conn_str (str): Database connection string.

    Returns:
        list: List of table names in the database.
    """
    engine = sql.create_engine(conn_str)
    inspect = sql.inspect(engine)
    table_names = inspect.get_table_names()
    return table_names

def db_read_raw_els_table(table_name, conn_str = CONN_STR): 
    """
    Read a raw table from the CRM database.

    Args:
        table_name (str): Name of the table to read.
        conn_str (str): Database connection string.

    Returns:
        pandas.DataFrame: The requested table as a DataFrame.
    """
    engine = sql.create_engine(conn_str)
    
    with engine.connect() as conn:
        df = pd.read_sql(
            sql.text(f"SELECT * FROM {table_name}"),  
            con=conn
        )
    
    return df

def db_read_els_data(conn_str = CONN_STR): 
    """
    Read and combine raw data from various tables in the CRM database.

    Args:
        conn_str (str): Database connection string.

    Returns:
        pandas.DataFrame: Combined and processed data from multiple tables.
    """
    engine = sql.create_engine(conn_str)
    
    with engine.connect() as conn:
        # Read Subscribers table
        subscribers_df = pd.read_sql(
            sql=text("SELECT * FROM Subscribers"),
            con = conn
        )
        
        # Convert data types
        subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
        subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')
        subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')
       
        # Read Tags table
        tags_df = pd.read_sql(
            sql=text("SELECT * FROM Tags"),
            con = conn
        )
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype('int')
    
        # Read Transactions table
        transactions_df = pd.read_sql(
            sql=text("SELECT * FROM Transactions"),
            con = conn
        )
     
        # Process user events
        user_events_df = tags_df \
            .groupby('mailchimp_id') \
            .agg(dict(tag = 'count')) \
            .set_axis(['tag_count'], axis=1) \
            .reset_index()
    
        # Join subscribers data with user events
        subscribers_joined_df = subscribers_df.merge(
            right = user_events_df,
            how = 'left',
            left_on = 'mailchimp_id',
            right_on = 'mailchimp_id')\
                .fillna(dict(tag_count = 0))
        subscribers_joined_df['tag_count'] = subscribers_joined_df['tag_count'].astype('int')
        
        # Add purchase information
        emails_made_purchase = transactions_df['user_email'].unique()
        subscribers_joined_df['made_purchase'] = subscribers_joined_df['user_email'] \
            .isin(emails_made_purchase) \
            .astype('int')
    return subscribers_joined_df    

def process_leads_tags(leads_df, tags_df):
    """
    Process the leads and tags data for machine learning models.

    Args:
        leads_df (pandas.DataFrame): Output from the els.db_read_els_data() function.
        tags_df (pandas.DataFrame): Output from the els.db_read_raw_els_table('tags') function.

    Returns:
        pandas.DataFrame: Processed leads and tags data combined for ML models.
    """
    # Calculate days since opt-in
    date_max = leads_df['optin_time'].max()
    leads_df['optin_days'] = (leads_df['optin_time'] - date_max).dt.days
    
    # Extract email provider
    leads_df['email_provider'] = leads_df['user_email'].map(lambda x: x.split('@')[1])
    
    # Calculate tag count per day
    leads_df['tag_count_by_optin_day'] = leads_df['tag_count'] / abs(leads_df['optin_days'] - 1)
    
    # Create wide format of tags
    tags_wide_leads_df = tags_df \
    .assign(value = lambda x: 1) \
    . pivot_table(
        index = 'mailchimp_id',
        columns = 'tag',
        values = 'value',
    ) \
    . fillna(value = 0) \
    . pipe (
        func = jn.clean_names
    )
    tags_wide_leads_df.columns = tags_wide_leads_df.columns \
    . to_series() \
    .apply(func = lambda x: f'tag_{x}') \
    . to_list()

    tags_wide_leads_df = tags_wide_leads_df.reset_index()

    # Merge leads and tags data
    leads_tags_df  = leads_df \
    .merge(tags_wide_leads_df, how = 'left')   
    
    def fillna_regex(data, regex, value = 0, **kwargs):
        """Fill NA for columns that match a regular expression"""
        for col in data.columns:
            if re.match(pattern = regex, string = col):
                data[col] = data[col].fillna(value= value, **kwargs)
        return data 

    leads_tags_df = fillna_regex(leads_tags_df, regex = 'tag_', value = 0)
            
    # Filter countries based on sales
    countries_to_keep = explore_sales_by_category(
        data = leads_tags_df,
        category = 'country_code'
    ) \
        .query('sales >= 6') \
        .index \
        .to_list()

    leads_tags_df['country_code'] = leads_tags_df['country_code'] \
    .apply(lambda x: x if x in countries_to_keep else 'Other')
    
    return leads_tags_df

def db_read_and_process_els_data(conn_str = CONN_STR):
    """
    Read and process email lead scoring data from the database.

    Args:
        conn_str (str): Database connection string.

    Returns:
        pandas.DataFrame: Processed leads and tags data ready for analysis.
    """
    leads_df = db_read_els_data(conn_str = conn_str)
    
    tags_df = db_read_raw_els_table(
        conn_str = conn_str,
        table_name = 'Tags'
    )
    
    df = process_leads_tags(leads_df, tags_df)
    
    return df 