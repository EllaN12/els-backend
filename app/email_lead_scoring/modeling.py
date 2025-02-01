import pandas as pd
import pycaret.classification as clf
import app.email_lead_scoring as els
import numpy as np
import mlflow

# Load the leads data
leads_df = els.db_read_and_process_els_data()

def model_score_leads(
    data,
    model_path = "app/models/blended_models_final"
):
    """
    Score leads using a pre-trained PyCaret model.

    This function takes lead data, loads a pre-trained model, and generates
    lead scores based on the model's predictions.

    Args:
        data (pandas.DataFrame): Lead data from email_lead_scoring.db_read_and_process_els_data().
        model_path (str): Path to the PyCaret model to load.

    Returns:
        pandas.DataFrame: Original lead data with added lead scores.

    Raises:
        ValueError: If the model output lacks required columns.
    """
    # Load the PyCaret model
    mod = clf.load_model(model_path)

    # Generate predictions
    predictions_df = clf.predict_model(estimator=mod, data=data)

    # Validate prediction output
    if 'prediction_score' not in predictions_df.columns:
        raise ValueError("The model prediction output lacks a 'prediction_score' column.")

    if 'prediction_label' not in predictions_df.columns:
        raise ValueError("The model prediction output lacks a 'prediction_label' column.")

    # Calculate lead scores
    predictions_df['lead_score'] = 1 - predictions_df['prediction_score']

    # Combine lead scores with original data
    leads_scored_df = pd.concat([predictions_df['lead_score'], data], axis=1)

    return leads_scored_df

def mlflow_get_best_run(
    experiment_name, n=1,
    metric= ["metrics.auc"],
    ascending=False,
    tag_source = ['finalize_model', 'H2O_AutoML_Model']
    ):
    """
    Retrieve the best run from an MLflow experiment based on specified metrics.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        n (int): Which n-th best run to retrieve (1 for best, 2 for second-best, etc.).
        metric (list): List of metrics to sort by.
        ascending (bool): Sort order for metrics (False for descending, True for ascending).
        tag_source (list): List of acceptable sources for the run.

    Returns:
        tuple: Best run ID and experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    logs_df = mlflow.search_runs(experiment_id)
    best_run_id = logs_df \
        .query(f"`tags.Source` in {tag_source}") \
        .sort_values(metric, ascending=ascending) \
        ["run_id"] \
        .values \
        [n-1]
    return best_run_id , experiment_id 

def mlflow_score_leads(data, run_id):
    """
    Score leads using a model from an MLflow run.

    This function loads a model from an MLflow run and uses it to score the provided lead data.

    Args:
        data (pandas.DataFrame): Lead data to be scored.
        run_id (str): MLflow run ID for the model to be used.

    Returns:
        pandas.DataFrame: Original lead data with added scores.
    """
    logged_model = f'runs:/{run_id[0]}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    # Generate predictions
    try:
        predictions_array = loaded_model.predict(pd.DataFrame(data))['p1']
    except:
        predictions_array = loaded_model._model_impl.predict(pd.DataFrame(data))['p1']
    
    predictions_series = pd.Series(predictions_array, name = "Score")
    
    # Combine scores with original data
    ret = pd.concat([predictions_series, data], axis = 1)
    
    return ret