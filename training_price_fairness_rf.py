import learning_curve
import train_model
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    # Load cleaned dataset using your existing GitHub function
    X, Y = train_model.load_dataset(filename='clean_apartment_data')

    # Random Forest parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True]
    }

    # Evaluate using your existing helper
    selected_model = train_model.model_evaluation(
        RandomForestRegressor(random_state=42),
        param_grid,
        X, Y
    )

    # Plot learning curve using your existing helper
    learning_curve.plot_learning_curve(
        selected_model.best_estimator_,
        X, Y,
        'Random_Forest_Price_Fairness_Analysis'
    )

    # Save model
    train_model.save_model(selected_model.best_estimator_, "RandomForest_rent_model")

    print("Random Forest training complete. Model saved to model/RandomForest_rent_model.pkl")
