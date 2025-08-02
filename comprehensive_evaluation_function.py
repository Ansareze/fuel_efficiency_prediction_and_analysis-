from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np

def evaluate_model_comprehensive(y_true, y_pred, model_name):
    """
    Comprehensive model evaluation function that calculates all common regression metrics.
    
    Parameters:
    y_true: Actual target values
    y_pred: Predicted target values
    model_name: Name of the model for display purposes
    
    Returns:
    dict: Dictionary containing all evaluation metrics
    """
    # Calculate basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape_score = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate Explained Variance Score
    ev_score = explained_variance_score(y_true, y_pred)
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.4f} MPG")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} MPG")
    print(f"R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"Mean Absolute Percentage Error (MAPE): {mape_score:.2f}%")
    print(f"Explained Variance Score: {ev_score:.4f}")
    
    # Return all metrics as a dictionary
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape_score,
        'explained_variance': ev_score
    }

# Example usage:
# results = evaluate_model_comprehensive(y_test, y_pred, "Linear Regression")
# 
# You can access individual metrics like:
# print(f"MAE: {results['mae']}")
# print(f"R²: {results['r2']}") 