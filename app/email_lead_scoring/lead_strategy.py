import pandas as pd
import numpy as np
import plotly.express as px
import app.email_lead_scoring as els



def lead_make_strategy(leads_scored_df, thresh = 0.99, for_marketing_team = False, verbose = False):
    """
    Create a lead strategy based on lead scores.

    This function categorizes leads into 'Hot-Lead' and 'Cold-Lead' based on a threshold.

    Args:
        leads_scored_df (pandas.DataFrame): DataFrame containing lead scores.
        thresh (float): Threshold for categorizing leads.
        for_marketing_team (bool): If True, includes additional data for marketing team use.
        verbose (bool): If True, prints progress messages.

    Returns:
        pandas.DataFrame: DataFrame with lead strategy categories.
    """
    leads_scored_small_df = leads_scored_df[['user_email', 'lead_score', 'made_purchase']]
    leads_ranked_df = leads_scored_small_df \
        .sort_values('lead_score', ascending = False) \
        .assign(rank = lambda x: np.arange(0, len(x['made_purchase'])) + 1 ) \
        .assign(gain = lambda x: np.cumsum(x['made_purchase']) / np.sum(x['made_purchase']))
        
    strategy_df = leads_ranked_df \
        .assign(category = lambda x: np.where(x['gain'] <= thresh, "Hot-Lead", "Cold-Lead"))
    
    if for_marketing_team:
        strategy_df = leads_scored_df \
            .merge(
                right       = strategy_df[['category']],
                how         = 'left',
                left_index  = True,
                right_index = True
            )
    if verbose:
        print("Strategy created")
    return strategy_df 


def lead_aggregate_strategy_results(strategy_df):
    """
    Aggregate results of the lead strategy.

    Args:
        strategy_df (pandas.DataFrame): DataFrame with lead strategy categories.

    Returns:
        pandas.DataFrame: Aggregated results by category.
    """
    results_df = strategy_df \
        .groupby('category') \
        .agg(
            count = ('made_purchase', 'count'),
            sum_made_purchase = ('made_purchase', 'sum')
        )
    
    return results_df


def lead_strategy_calc_expected_value(
    results_df,
    email_list_size = 1e5,
    unsub_rate_per_sales_email = 0.001,
    sales_emails_per_month = 5,
    avg_sales_per_month = 250000,
    avg_sales_emails_per_month = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    verbose = False,
):
    """
    Calculate the expected value of the lead strategy.

    This function estimates the financial impact of the lead strategy based on
    various marketing and sales parameters.

    Args:
        results_df (pandas.DataFrame): Aggregated results from lead_aggregate_strategy_results.
        email_list_size (float): Total size of the email list.
        unsub_rate_per_sales_email (float): Unsubscribe rate per sales email.
        sales_emails_per_month (int): Number of sales emails sent per month.
        avg_sales_per_month (float): Average sales per month.
        avg_sales_emails_per_month (int): Average number of sales emails per month.
        customer_conversion_rate (float): Rate at which leads convert to customers.
        avg_customer_value (float): Average value of a customer.
        verbose (bool): If True, prints detailed results.

    Returns:
        dict: Dictionary containing expected value, savings, and other metrics.
    """
    # Extract values from results_df, handling potential KeyErrors
    try:
        cold_lead_count = results_df['count']['Cold-Lead']
    except KeyError:
        cold_lead_count = 0

    try:
        hot_lead_count = results_df['count']['Hot-Lead']
    except KeyError:
        hot_lead_count = 0

    try:
        missed_purchases = results_df['sum_made_purchase']['Cold-Lead']
    except KeyError:
        missed_purchases = 0
        
    try:
        made_purchases = results_df['sum_made_purchase']['Hot-Lead']
    except KeyError:
        made_purchases = 0
        
    total_count = (cold_lead_count + hot_lead_count)
    
    total_purchases = (missed_purchases + made_purchases)
    sample_factor = email_list_size / total_count

    sales_per_email_sent = avg_sales_per_month / avg_sales_emails_per_month
    
    # Calculate savings and costs
    savings_cold_no_target = cold_lead_count * \
        sales_emails_per_month * unsub_rate_per_sales_email * \
        customer_conversion_rate * avg_customer_value * \
        sample_factor
        
    missed_purchase_ratio = missed_purchases / (missed_purchases + made_purchases)
    cost_missed_purchases = sales_per_email_sent * sales_emails_per_month * missed_purchase_ratio

    cost_hot_target_but_unsub = hot_lead_count * \
        sales_emails_per_month * unsub_rate_per_sales_email * \
        customer_conversion_rate * avg_customer_value * \
        sample_factor
    
    made_purchase_ratio = made_purchases / (missed_purchases + made_purchases)
    savings_made_purchases = sales_per_email_sent * sales_emails_per_month * made_purchase_ratio
    
    # Calculate expected value and savings
    ev = savings_made_purchases + savings_cold_no_target - cost_missed_purchases
    es = savings_cold_no_target - cost_missed_purchases
    esc = savings_cold_no_target / avg_customer_value
    
    if verbose:
        print(f"Expected Value: {'${:,.0f}'.format(ev)}")
        print(f"Expected Savings: {'${:,.0f}'.format(es)}")
        print(f"Monthly Sales: {'${:,.0f}'.format(savings_made_purchases)}")
        print(f"Saved Customers: {'{:,.0f}'.format(esc)}")
        
    return {
        'expected_value': ev,
        'expected_savings': es,
        'monthly_sales': savings_made_purchases,
        'expected_customers_saved': esc
    }
    
    
    
def lead_strategy_create_thresh_table(
    leads_scored_df,
    thresh = np.linspace(0, 1, num=100),
    email_list_size = 1e5,
    unsub_rate_per_sales_email = 0.005,
    sales_emails_per_month = 5,
    avg_sales_per_month = 250000,
    avg_sales_emails_per_month = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    highlight_max = True,
    highlight_max_color = "yellow",
    verbose = False,
):
    thresh_df = pd.Series(thresh, name = "thresh").to_frame()
    sim_results_list = [
        lead_make_strategy(leads_scored_df, thresh = tup[0], verbose = verbose) \
        .pipe(lead_aggregate_strategy_results) \
        .pipe(lead_strategy_calc_expected_value,
            email_list_size = email_list_size,
            unsub_rate_per_sales_email = unsub_rate_per_sales_email,
            sales_emails_per_month = sales_emails_per_month,
            avg_sales_per_month = avg_sales_per_month,
            avg_sales_emails_per_month = avg_sales_emails_per_month,
            customer_conversion_rate = customer_conversion_rate,
            avg_customer_value = avg_customer_value,
            verbose = verbose
        )
    for tup in zip(thresh_df['thresh'])
]   

    sim_results_df = pd.Series(sim_results_list, name = "sim_results").to_frame()

    sim_results_df = sim_results_df['sim_results'].apply(pd.Series)

    thresh_optim_df = pd.concat([thresh_df, sim_results_df], axis=1)


    if highlight_max:
        thresh_optim_df = thresh_optim_df.style.highlight_max(
            color = highlight_max_color,
            axis  = 0
        )

    return thresh_optim_df

    
    

def lead_select_optimum_thresh(
    thresh_optim_df,
    optim_col = "expected_value",
    monthly_sales_reduction_safe_guard = 0.90,
    verbose = False):
    
    # # Handle styler object
    
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df


    #Find optim
    _filter_1 = thresh_optim_df[optim_col] == thresh_optim_df[optim_col].max()

    # Find safegaurd
    _filter_2 = thresh_optim_df['monthly_sales'] >= monthly_sales_reduction_safe_guard * thresh_optim_df['monthly_sales'].max()
    
    # Test i optim is in the safegaurd
    if (all(_filter_1 & _filter_2 == _filter_2 )):
        _filter = _filter_1
    else:
        _filter = _filter_2
    
    # Apply filter
    thresh_selected = thresh_optim_df[_filter].head(1)
    
    # values
    ret = thresh_selected['thresh'].values[0]
    
    if verbose:
        print(f"Optimal Threshold: {ret}")
    return ret
        

def lead_get_expected_value(
    thresh_optim_df,
    threshold = 0.85,
    verbose = False
):

# Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
        
    df = thresh_optim_df[thresh_optim_df['thresh'] >= threshold]
    
    if verbose:
        print(df)
    return df

def lead_plot_optim_thresh(
    thresh_optim_df,
    optim_col = "expected_value",
    monthly_sales_reduction_safeguard = 0.90,
    verbose = False   
):
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
        
    # Make the plog
    fig = px.line(
        thresh_optim_df, 
        x = 'thresh',
        y = optim_col
    )

    fig.add_hline(y = 0, line_color = 'black')

    fig.add_vline(
        x = lead_select_optimum_thresh(
            thresh_optim_df, 
            optim_col=optim_col,    
            monthly_sales_reduction_safe_guard=monthly_sales_reduction_safeguard
        ),
        line_color = "red",
        line_dash  = "dash"
    )
    
    if verbose:
        print("Plot created.")
        
    return fig
    

def lead_score_strategy_optimization(
    leads_scored_df,
    thresh = np.linspace(0, 1, num=100),
    optim_col = "expected_value",
    monthly_sales_reduction_safe_guard = 0.90,
    for_marketing_team = True,
    email_list_size = 1e5,
    unsub_rate_per_sales_email = 0.005,
    sales_emails_per_month = 5,
    avg_sales_per_month = 250000,
    avg_sales_emails_per_month = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    highlight_max = True,
    highlight_max_color = "yellow",
    
    verbose = False,
):
    """_Leads Score Strategy Optimization_ Function
    that returns the following documents:
    - lead_strategy_df
    - expected_value
    - thresh_optim_df
    - thresh_plot
    

    Args:
        leads_scored_df (_DataFrame_): _output of els.model_score_leads(leads_df)_.
        thresh (_Numpy Array_, optional): _the threshold to optimize_. Defaults to np.linspace(0, 1, num=100).
        optim_col (str, optional): _Optimization column  from the Strategy Table_. Defaults to "expected_value".
        monthly_sales_reduction_safe_guard (float, optional): _Management Risk Tolerance for reduced sales in month 1 as a percentage of total sales_. Defaults to 0.90.
        for_marketing_team (bool, optional): _produces the leads_strategy_df as an output to Marketing. Defaults to True.
        email_list_size (_type_, optional): _email size_. Defaults to 1e5.
        unsub_rate_per_sales_email (float, optional): _ Unsubscribe rate given by Marketing_. Defaults to 0.005.
        sales_emails_per_month (int, optional): _Number of sales emails blasted every month _. Defaults to 5.
        avg_sales_per_month (int, optional): _approximate sales expectation per month_. Defaults to 250000.
        avg_sales_emails_per_month (int, optional): _number of sales email blasts per month_. Defaults to 5.
        customer_conversion_rate (float, optional): _number of customers that converts on a monthly basis_. Defaults to 0.05.
        avg_customer_value (int, optional): _description_. Defaults to 2000.
        highlight_max (bool, optional): _Should we highlight the _. Defaults to True.
        highlight_max_color (str, optional): _color for highlights_. Defaults to "yellow".
        verbose (bool, optional): _should we print the progress statement_. Defaults to False.
    """
    # Lead strategy create thresh table
    thresh_optim_df = lead_strategy_create_thresh_table(
        leads_scored_df=leads_scored_df,
        thresh=thresh,
        email_list_size=email_list_size,
        unsub_rate_per_sales_email = unsub_rate_per_sales_email,
        sales_emails_per_month = sales_emails_per_month,
        avg_sales_per_month = avg_sales_per_month,
        avg_sales_emails_per_month = avg_sales_emails_per_month,
        customer_conversion_rate = customer_conversion_rate,
        avg_customer_value = avg_customer_value,
        highlight_max = highlight_max,
        highlight_max_color = highlight_max_color,
        verbose=verbose
    )
    
    # Lead Select Optimum Thresh
    thresh_optim = lead_select_optimum_thresh(
        thresh_optim_df,
        optim_col = optim_col,
        monthly_sales_reduction_safe_guard = monthly_sales_reduction_safe_guard,
        verbose = verbose
    )
    
    # Expected value
    expected_value = lead_get_expected_value(
        thresh_optim_df, 
        threshold = thresh_optim, 
        verbose = verbose
    )
    # Lead plot
    thresh_plot = lead_plot_optim_thresh(
        thresh_optim_df,
        optim_col = optim_col,
        monthly_sales_reduction_safeguard = monthly_sales_reduction_safe_guard,
        verbose = verbose
    )
    
    # Recalculate Lead Strategy
    lead_strategy_df = lead_make_strategy(
        leads_scored_df, 
        thresh             = thresh_optim, 
        for_marketing_team = for_marketing_team, 
        verbose            = verbose
    )    
    # Dictionary for return
    
    ret = dict(
        lead_strategy_df = lead_strategy_df,
        expected_value   = expected_value,
        thresh_optim_df  = thresh_optim_df,
        thresh_plot      = thresh_plot
    )
    
    return(ret)


