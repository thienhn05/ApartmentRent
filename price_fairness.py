def classify_price_fairness(actual_rent, predicted_rent, threshold_percent=10):
    """
    Compare actual rent with predicted fair rent.

    Parameters:
        actual_rent (float): Actual listing rent
        predicted_rent (float): Predicted fair rent from model
        threshold_percent (float): Threshold for fairness classification

    Returns:
        dict: difference, percentage_difference, and status
    """
    if predicted_rent <= 0:
        return {
            "difference": 0.0,
            "percentage_difference": 0.0,
            "status": "Invalid Prediction"
        }

    difference = actual_rent - predicted_rent
    percentage_difference = (difference / predicted_rent) * 100

    if percentage_difference > threshold_percent:
        status = "Overpriced"
    elif percentage_difference < -threshold_percent:
        status = "Underpriced"
    else:
        status = "Fairly Priced"

    return {
        "difference": round(difference, 2),
        "percentage_difference": round(percentage_difference, 2),
        "status": status
    }
