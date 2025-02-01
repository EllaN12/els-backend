# Import necessary libraries and modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
from uvicorn import Config, Server
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import app.email_lead_scoring as els

# Initialize FastAPI application
app = FastAPI()

# Set the application title
app.title = "Email Lead Scoring API"

# Configure CORS settings
ALLOWED_ORIGIN = "*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and process lead data from the database
leads_df = els.db_read_and_process_els_data()

@app.get("/")
async def main():
    """
    Render the main page of the application.
    
    Returns:
    HTMLResponse: A styled HTML page with a welcome message and link to API docs.
    """
    # HTML content with inline CSS for styling and animations
    content = """
    <head>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideIn {
                from { transform: translateY(-20px); }
                to { transform: translateY(0); }
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        </style>
    </head>
    <body style="font-family: 'Arial', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh;">
        <div style="background-color: rgba(255, 255, 255, 0.9); border-radius: 12px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); padding: 50px; max-width: 600px; text-align: center; animation: fadeIn 1s ease-out, slideIn 0.5s ease-out;">
            <h1 style="color: #4a69bd; font-size: 2.8em; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); animation: pulse 2s infinite;">Welcome to the Email Lead Scoring Project</h1>
            <p style="color: #34495e; font-size: 1.2em; line-height: 1.8; margin-bottom: 25px; animation: fadeIn 1.5s ease-out;">This API empowers users to score leads using our cutting-edge, proprietary lead scoring models.</p>
            <p style="color: #34495e; font-size: 1.2em; line-height: 1.8; margin-bottom: 35px; animation: fadeIn 2s ease-out;">Explore our API documentation by visiting <code style="background-color: #f1c40f; padding: 5px 8px; border-radius: 6px; color: #2c3e50; font-weight: bold;">/docs</code>.</p>
            <a href="/docs" style="display: inline-block; background: linear-gradient(45deg, #3498db, #2980b9); color: white; text-decoration: none; padding: 15px 30px; border-radius: 50px; font-weight: bold; font-size: 1.1em; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.2); animation: pulse 2s infinite;">View API Docs</a>
        </div>
    </body>
    """
    
    return HTMLResponse(content=content)

@app.get("/get_email_subscribers")
async def get_email_subscribers():
    """
    Retrieve all email subscribers from the database.
    
    Returns:
    JSONResponse: A JSON representation of the leads dataframe.
    """
    leads_json = leads_df.to_json()
    return JSONResponse(leads_json)

@app.post("/data")
async def data(request: Request):
    """
    Process incoming JSON data and return it as a JSON response.
    
    Args:
    request (Request): The incoming request containing JSON data.
    
    Returns:
    JSONResponse: A JSON representation of the processed data.
    """
    request_body = await request.body()
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json)
    leads_json = leads_df.to_json()
    return JSONResponse(leads_json)

@app.post("/predict")
async def predict(request: Request):
    """
    Score leads using the pre-trained model.
    
    Args:
    request (Request): The incoming request containing lead data in JSON format.
    
    Returns:
    JSONResponse: A JSON object containing the lead scores.
    """
    request_body = await request.body()
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json) 
    
    leads_scored_df = els.model_score_leads(
        data=leads_df,
        model_path="models/xgb_model_tuned"
    )

    scores = leads_scored_df[['lead_score']].to_dict()
    
    return JSONResponse(scores)

@app.post("/calculate_lead_strategy")
async def calculate_lead_strategy(
    request: Request,
    monthly_sales_reduction_safe_guard: float = 0.9,
    email_list_size: int = 100000,
    unsub_rate_per_sales_email: float = 0.005,
    sales_emails_per_month: int = 5,
    avg_sales_per_month: float = 250000.0,
    avg_sales_emails_per_month: int = 5,
    customer_conversion_rate: float = 0.05,
    avg_customer_value: float = 2000.0,
):
    """
    Calculate the optimal lead strategy based on various parameters and incoming lead data.
    
    Args:
    request (Request): The incoming request containing lead data in JSON format.
    monthly_sales_reduction_safe_guard (float): Safety factor for monthly sales reduction.
    email_list_size (int): Total size of the email list.
    unsub_rate_per_sales_email (float): Unsubscribe rate per sales email.
    sales_emails_per_month (int): Number of sales emails sent per month.
    avg_sales_per_month (float): Average sales per month.
    avg_sales_emails_per_month (int): Average number of sales emails per month.
    customer_conversion_rate (float): Rate at which leads convert to customers.
    avg_customer_value (float): Average value of a customer.
    
    Returns:
    JSONResponse: A JSON object containing the lead strategy, expected value, and threshold optimization table.
    """
    request_body = await request.body()
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json) 
    leads_scored_df = els.model_score_leads(
        data=leads_df,
        model_path="app/models/xgb_model_tuned"
    )
    optimization_results = els.lead_score_strategy_optimization(
        leads_scored_df=leads_scored_df,
        monthly_sales_reduction_safe_guard=monthly_sales_reduction_safe_guard,
        email_list_size=email_list_size,
        unsub_rate_per_sales_email=unsub_rate_per_sales_email,
        sales_emails_per_month=sales_emails_per_month,
        avg_sales_per_month=avg_sales_per_month,
        avg_sales_emails_per_month=avg_sales_emails_per_month,
        customer_conversion_rate=customer_conversion_rate,
        avg_customer_value=avg_customer_value,
    )
    results = {
        'lead_strategy': optimization_results['lead_strategy_df'].to_json(),
        'expected_value': optimization_results['expected_value'].to_json(),
        'thresh_optim_table': optimization_results['thresh_optim_df'].data.to_json()
    }
    
    return JSONResponse(results)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    config = Config(app=app, host="0.0.0.0", port=port, loop="asyncio")
    server = Server(config)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.serve())