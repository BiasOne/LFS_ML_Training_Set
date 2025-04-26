import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model types
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def lin_reg(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_lr_train_pred = lr.predict(x_train)
    y_lr_test_pred = lr.predict(x_test)
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_rmse = np.sqrt(lr_train_mse)
    lr_train_mae = mean_absolute_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_rmse = np.sqrt(lr_test_mse)
    lr_test_mae = mean_absolute_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)
    print_stats("Linear Regression", lr_train_mse, lr_train_rmse, lr_train_mae, lr_train_r2, lr_test_mse, lr_test_rmse, lr_test_mae, lr_test_r2)
    return y_lr_test_pred

def rand_forest_reg(x_train, x_test, y_train, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    y_rfr_train_pred = rfr.predict(x_train)
    y_rfr_test_pred = rfr.predict(x_test)
    y_rfr_train_pred
    rfr_train_mse = mean_squared_error(y_train, y_rfr_train_pred)
    rfr_train_rmse = np.sqrt(rfr_train_mse)
    rfr_train_mae = mean_absolute_error(y_train, y_rfr_train_pred)
    rfr_train_r2 = r2_score(y_train, y_rfr_train_pred)
    rfr_test_mse = mean_squared_error(y_test, y_rfr_test_pred)
    rfr_test_rmse = np.sqrt(rfr_test_mse)
    rfr_test_mae = mean_absolute_error(y_test, y_rfr_test_pred)
    rfr_test_r2 = r2_score(y_test, y_rfr_test_pred)    
    print_stats("Random Forest Regressor", rfr_train_mse, rfr_train_rmse, rfr_train_mae, rfr_train_r2, rfr_test_mse, rfr_test_rmse, rfr_test_mae, rfr_test_r2)
    return y_rfr_test_pred

def dec_tree_reg(x_train, x_test, y_train, y_test):
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    y_dt_train_pred = dt.predict(x_train)
    y_dt_test_pred = dt.predict(x_test)
    y_dt_train_pred
    dt_train_mse = mean_squared_error(y_train, y_dt_train_pred)
    dt_train_rmse = np.sqrt(dt_train_mse)
    dt_train_mae = mean_absolute_error(y_train, y_dt_train_pred)
    dt_train_r2 = r2_score(y_train, y_dt_train_pred)
    dt_test_mse = mean_squared_error(y_test, y_dt_test_pred)
    dt_test_rmse = np.sqrt(dt_test_mse)
    dt_test_mae = mean_absolute_error(y_test, y_dt_test_pred)
    dt_test_r2 = r2_score(y_test, y_dt_test_pred)
    print_stats("Decision Tree Regressor", dt_train_mse, dt_train_rmse, dt_train_mae, dt_train_r2, dt_test_mse, dt_test_rmse, dt_test_mae, dt_test_r2)
    return y_dt_test_pred    
    
def print_stats(model, train_mse, train_rmse, train_mae, train_r2, test_mse, test_rmse, test_mae, test_r2):
    print(f"****** {model} ******")
    print(f"Training Set Mean Squared Error: {train_mse:.4f}")
    print(f"Training Set Root Mean Squared Error: {train_rmse:.4f}")
    print(f"Training Set Mean Absolute Error: {train_mae:.4f}")
    print(f"Training Set R2: {train_r2:.4f}")
    print(f"Test Set Mean Squared Error: {test_mse:.4f}")
    print(f"Test Set Root Mean Squared Error: {test_rmse:.4f}")
    print(f"Test Set Mean Absolute Error: {test_mae:.4f}")
    print(f"Test Set R2: {test_r2:.4f}")
    print("-" * (len(model) + 14)) 

def plot_regression_results(y_test, predictions, model_names):
    plot_df = pd.DataFrame({
        'Actual': y_test.squeeze()
    })

    for name, preds in zip(model_names, predictions):
        plot_df[name] = preds
        plot_df[f'{name}_Residuals'] = plot_df['Actual'] - plot_df[name]
        print(plot_df)

    plt.figure(figsize=(15, 5))
    for i, name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)
        sns.scatterplot(x='Actual', y=name, data=plot_df, alpha=0.6)
        plt.title(f'{name}: Actual vs. Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        min_val = min(plot_df['Actual'].min(), plot_df[name].min())
        max_val = max(plot_df['Actual'].max(), plot_df[name].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    for i, name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)
        sns.scatterplot(x=name, y=f'{name}_Residuals', data=plot_df, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.6)
        plt.title(f'{name}: Residuals vs. Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    for i, name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)
        sns.histplot(plot_df[f'{name}_Residuals'], kde=True)
        plt.title(f'{name}: Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def main():    
    # df = pd.read_csv('D:/Projects/File_systems/code/training_dataset_viral_only.csv')
    df = pd.read_csv('D:/Projects/File_systems/code/training_dataset.csv')
    y = df['optimal_promotion_day']
    x = df.drop(['object_id', 'object_type', 'days_until_optimal_promotion', 'spike_day', 'trend_start_day', 'optimal_promotion_day' ], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)    
    lr_test_pred = lin_reg(x_train, x_test, y_train, y_test)
    rfr_test_pred = rand_forest_reg(x_train, x_test, y_train, y_test)
    dt_test_pred = dec_tree_reg(x_train, x_test, y_train, y_test)

    test_predictions = [lr_test_pred, rfr_test_pred, dt_test_pred]
    model_names = ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]

    plot_regression_results(y_test, test_predictions, model_names)


if __name__ == '__main__':
    main()