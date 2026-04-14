import learning_curve
import train_model
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    # 1. Load the cleaned Apartment Rent dataset [5]
    # This assumes load_dataset() in train_model.py reads your CSV from /dataset folder
    X, Y = train_model.load_dataset(filename='clean_apartment_data')

    # 2. Define the parameter grid for tuning based on assignment specs [6]
    # These parameters control tree complexity to balance bias and variance [7]
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  
        'max_depth': [1, 8, 9], # Controls complexity and prevents overfitting [6]
        'min_samples_split': [1, 9, 10], # Min samples needed to split an internal node [6]
        'min_samples_leaf': [1, 9, 11], # Min samples required to be at a leaf node [6]
        'max_features': [None, 'sqrt', 'log2'], # Number of features considered for best split [12]
    }

    # 3. Evaluate the model using GridSearchCV and 5-fold Cross-Validation [3]
    # This function returns the best performing estimator based on RMSE/R² [13]
    selected_model = train_model.model_evaluation(
        DecisionTreeRegressor(random_state=42), 
        param_grid, 
        X, Y
    )

    # 4. Plot the learning curve for Overfitting Analysis [4, 14]
    # This visualizes how the Decision Tree generalizes as training samples increase [15]
    learning_curve.plot_learning_curve(
        selected_model.best_estimator_, 
        X, Y, 
        'Decision_Tree_Apartment_Analysis'
    )

    # 5. Save the selected model for the Streamlit Standalone Framework [16, 17]
    # This creates the .pkl file that your app.py will load
    train_model.save_model(selected_model.best_estimator_, "DecisionTree_rent_model")

    print("Decision Tree training complete. Model saved to model/DecisionTree_rent_model.pkl")
