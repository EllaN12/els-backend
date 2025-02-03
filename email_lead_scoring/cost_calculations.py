import pandas as pd
import numpy as np
import janitor
import plotly.express as px
import pandas_flavor as pf

@pf.register_dataframe_method
def cost_calc_monthly_cost_table(
    email_list_size = 1e5,
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12
):
    """
    Calculate the monthly cost table for email marketing campaigns.

    This function generates a DataFrame with projections of email list size,
    lost customers, and associated costs over a specified number of periods.

    Args:
        email_list_size (float): Initial size of the email list.
        email_list_growth_rate (float): Monthly growth rate of the email list.
        sales_emails_per_month (int): Number of sales emails sent per month.
        unsub_rate_per_sales_email (float): Unsubscription rate per sales email.
        customer_conversion_rate (float): Rate at which email subscribers convert to customers.
        average_customer_value (float): Average value of a customer.
        n_periods (int): Number of months to project.

    Returns:
        pandas.DataFrame: A DataFrame containing projections for each period.
    """
    # Create a series for the periods
    period_series = pd.Series(np.arange(0, n_periods), name = "Period") 
    cost_table2_df = period_series.to_frame()
    
    # Calculate email list size without growth
    cost_table2_df["Email_List_Size_No_Growth"] = np.repeat(email_list_size, n_periods)
    
    # Calculate lost customers without growth
    cost_table2_df["Lost_Customers_No_Growth"] = cost_table2_df['Email_List_Size_No_Growth'] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate
    
    # Calculate cost without growth
    cost_table2_df["Cost_No_Growth"] = cost_table2_df['Lost_Customers_No_Growth'] * average_customer_value
    
    # Calculate email list size with growth
    cost_table2_df["Email_List_Size_Growth"] = cost_table2_df['Email_List_Size_No_Growth'] * ((1 + email_list_growth_rate)**cost_table2_df['Period']) 

    # Calculate lost customers with growth
    cost_table2_df["Lost_Customers_Growth"] = cost_table2_df['Email_List_Size_Growth'] * unsub_rate_per_sales_email * sales_emails_per_month * customer_conversion_rate
    
    # Calculate cost with growth
    cost_table2_df["Cost_With_Growth"] = cost_table2_df['Lost_Customers_Growth'] * average_customer_value 
    
    return cost_table2_df

def cost_total_unsub_cost(cost_table2_df):
    """
    Calculate the total unsubscription cost from the monthly cost table.

    Args:
        cost_table2_df (pandas.DataFrame): The output from cost_calc_monthly_cost_table().

    Returns:
        pandas.DataFrame: A summary of total costs with and without growth.
    """
    summary_df =  cost_table2_df[["Cost_No_Growth", "Cost_With_Growth"]] \
        .sum()\
        .to_frame() \
        .transpose()
    return summary_df

def cost_simulate_unsub_cost(
    email_list_monthly_growth_rate = [0,0.035],
    customer_conversion_rate = [0.04, 0.05,0.06],
    **kwargs
):
    """
    Simulate unsubscription costs for different growth rates and conversion rates.

    This function generates a cost simulation for various combinations of email list
    growth rates and customer conversion rates to model cost uncertainty.

    Args:
        email_list_monthly_growth_rate (list): List of email monthly growth rates to simulate.
        customer_conversion_rate (list): List of customer conversion rates to simulate.
        **kwargs: Additional parameters to pass to cost_calc_monthly_cost_table().

    Returns:
        pandas.DataFrame: Simulation results for different parameter combinations.
    """
    # Create parameter grid
    data_dict = dict(
    email_list_monthly_growth_rate = email_list_monthly_growth_rate,
    customer_conversion_rate = customer_conversion_rate)
    import itertools

    def cartesian_product(data_dict):
        keys = data_dict.keys()
        values = data_dict.values()
        combinations = list(itertools.product(*values))
        return pd.DataFrame(combinations, columns=keys)

    parameter_grid_df = cartesian_product(data_dict)
    
    # Define temporary function for cost calculation
    def temporary_function(x,y):
        cost_table_df = cost_calc_monthly_cost_table(
            email_list_growth_rate = x,
            customer_conversion_rate = y,
            **kwargs
        )
        cost_summary_df = cost_total_unsub_cost(cost_table_df)
        return cost_summary_df
    
    # Simulate costs using list comprehension
    summary_list = [temporary_function(x, y) for x, y in zip(parameter_grid_df['email_list_monthly_growth_rate'], parameter_grid_df['customer_conversion_rate'])]

    simulation_results_df = pd.concat(summary_list, axis=0)\
        .reset_index() \
        .drop("index", axis = 1)\
        .merge(parameter_grid_df.reset_index(), left_index = True, right_index = True)
    
    return simulation_results_df

@pf.register_series_method
def cost_plot_simulated_unsub_costs(simulation_results):
    """
    Plot the simulated unsubscription costs.

    This function creates a heatmap visualization of the simulation results,
    showing how costs vary with different growth rates and conversion rates.

    Args:
        simulation_results (pandas.DataFrame): The output from cost_simulate_unsub_cost().

    Returns:
        plotly.graph_objs._figure.Figure: A heatmap of the simulation results.
    """
    simulation_results_wide_df = simulation_results\
        .drop('Cost_No_Growth', axis = 1)\
        .pivot(
            index = 'email_list_monthly_growth_rate',
            columns = 'customer_conversion_rate',
            values = 'Cost_With_Growth'
        )
    
    fig = px.imshow(
        simulation_results_wide_df,
        origin='lower',
        aspect='auto',
        title='Lead Cost Simulation',
        labels=dict(
            x ='Customer Conversion Rate',
            y ='Monthly Email List Growth Rate',
            color='Cost of Unsubscription'
        )   
    )
    
    return fig

# Example usage of the cost simulation and plotting functions
cost_simulate_unsub_cost(
    email_list_monthly_growth_rate=[0.01, 0.02, 0.03],
    customer_conversion_rate=[0.04, 0.05, 0.06],
    email_list_size=100000)\
. pipe(cost_plot_simulated_unsub_costs)