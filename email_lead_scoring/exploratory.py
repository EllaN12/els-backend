import pandas as pd
import numpy as np

# Function: Explore Sales By Category  

def explore_sales_by_category(data, category = 'country_code', sort_by = ['sales', 'prop_in_group' ]):
    """
    Explore sales data by a specified category.

    This function analyzes sales data grouped by a given category, calculating various
    proportions and cumulative sums.

    Args:
        data (pandas.DataFrame): Subscribers data generated from db_read_els_data() function.
        category (str): Categorical column from the subscribers data (e.g., 'country_code', 'member_rating').
        sort_by (list): Methods to sort the table (e.g., by sales amount or proportion in group).

    Returns:
        pandas.DataFrame: A DataFrame with sales analysis by the specified category.
    """
    # Handle sort_by parameter
    if isinstance(sort_by, list):
        sort_by = sort_by[0]
    
    # Data Manipulation
    ret = data \
        .groupby(category) \
        .agg(dict(made_purchase = ['sum', lambda x: sum(x) / len(x)])) \
        .set_axis(['sales', 'prop_in_group'], axis=1) \
        .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])) \
        .sort_values(by= sort_by, ascending=False) \
        .assign(prop_cumsum = lambda x: x['prop_overall'].cumsum()) 
        
    return ret

# Function: Explore Sales by Numeric Feature
def explore_sales_by_numeric(
    data,
    numeric = 'tag_count',
    q = [0.10, 0.50, 0.90]
):
    """
    Explore sales data using the 'made_purchase' column and specified numeric column(s).

    This function analyzes the relationship between purchases and numeric features
    in the subscriber data.

    Args:
        data (pandas.DataFrame): Subscribers data generated from db_read_els_data() function.
        numeric (str or list): Numeric column(s) to analyze against purchases.
        q (list): Quantiles to calculate for the numeric columns.

    Returns:
        pandas.DataFrame: A DataFrame with quantile analysis of numeric features grouped by purchase status.
    """
    if isinstance(numeric, list):
        feature_list = ['made_purchase', *numeric ]
    else:
        feature_list = ['made_purchase', numeric]
        
    ret = data[feature_list] \
        .groupby('made_purchase') \
        .quantile(q=q, numeric_only=True)
        
    return ret