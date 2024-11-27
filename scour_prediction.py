# Import necessary libraries for data processing, modeling, and visualization
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import csv
import shap

## =========================== Part 1: Loading and Preprocessing Data ===========================

# Load the dataset from a CSV file. Ensure the file is present in the same directory as the script.
scour = pd.read_csv("Scour_Field.csv", names=['b', 'V0', 'Vc', 'y0', 'd50', 'ys'])

# Description of columns in the dataset:
# 'b': Pier width (ft)
# 'V0': Mean flow velocity (ft/s)
# 'Vc': Critical velocity (ft/s)
# 'y0': Upstream flow depth (ft)
# 'd50': Median particle size (mm)
# 'ys': Observed scour depth (ft)

# Compute the correlation matrix to identify relationships between variables
corr_matrix = scour.corr()

# Visualize the correlation matrix using a heatmap
f, ax = plt.subplots(figsize=(12, 8))  # Set the figure size
sns.set(style="white")
cbar_kws = {"shrink": 1}  # Shrink the color bar for better aesthetics
g = sns.heatmap(
    corr_matrix,
    mask=np.triu(np.ones_like(corr_matrix, dtype=np.bool_)),  # Mask the upper triangle of the matrix
    cmap=sns.diverging_palette(240, 1, as_cmap=True),
    square=True,
    annot=True,
    annot_kws={"size": 22},
    ax=ax,
    cbar_kws=cbar_kws
)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.savefig("High_resolution_dimension.png", dpi=600)

# Adjust the color bar font size for readability
colorbar = g.collections[0].colorbar
colorbar.ax.tick_params(labelsize=22)
plt.show()

# Create new features (dimensionless variables) for model input
scour[r'$y/b$'] = scour['y0'] / scour['b']  # Flow depth to pier width ratio
scour[r'$b/d_50$'] = (scour['b'] / 3.28) / (scour['d50'] / 1000)  # Pier width to particle size ratio
scour[r'$V/V_c$'] = scour['V0'] / scour['Vc']  # Velocity to critical velocity ratio
scour[r'$Fr_b$'] = scour['V0'] / (32.2 * scour['b']) ** 0.5  # Froude number
scour[r'$ys/b$'] = scour['ys'] / scour['b']  # Observed scour depth to pier width ratio

# Define the input (independent variables) and output (dependent variable) for modeling
input_data = scour[[r'$y/b$', r'$b/d_50$', r'$V/V_c$', r'$Fr_b$']]
output_data = scour[r'$ys/b$']

# Scale the input data to normalize the range between 0 and 1
scaler = MinMaxScaler()
scaled_input = scaler.fit_transform(input_data)
scaled_output = output_data.values  # Output data is not scaled

## ====================== Part 2: Splitting Data ======================

# Split the data into training (80%), validation (10%), and test (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(scaled_input, scaled_output, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1004)

## ====================== Part 3: Model Training ======================

# Define hyperparameters for the XGBoost model
params = {
    'objective': 'reg:squarederror',  # Objective for regression
    'n_estimators': 1000,  # Maximum number of boosting iterations
    'learning_rate': 0.02,  # Learning rate for gradient boosting
    'gamma': 0.7,  # Minimum loss reduction for a split
    'subsample': 0.58,  # Fraction of data used for training per iteration
    'max_depth': 5,  # Maximum depth of trees
    'min_child_weight': 3,  # Minimum sum of weights for child nodes
    'reg_alpha': 0.6,  # L1 regularization term
    'reg_lambda': 0.5  # L2 regularization term
}

# Initialize and train the XGBoost regressor
xgb_model = xgb.XGBRegressor(**params)
xgb_model.fit(
    X_train, y_train,
    early_stopping_rounds=50,  # Stop training if no improvement in validation error for 50 rounds
    eval_set=[(X_val, y_val)],
    verbose=False
)

## ====================== Part 4: Metrics and Analysis ======================

# Define a function to calculate various evaluation metrics
def calculate_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
    mean_obs = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    nmse = np.mean((y_pred - y_true) ** 2) / (mean_obs * mean_pred)  # Normalized Mean Squared Error
    ia = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((np.abs(y_true - mean_obs) + np.abs(y_pred - mean_obs)) ** 2)  # Index of Agreement
    si = rmse / mean_obs  # Scatter Index
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - mean_obs) ** 2)  # Nash-Sutcliffe Efficiency
    r2 = (np.corrcoef(y_true, y_pred)[0, 1]) ** 2  # Coefficient of determination
    bias = np.mean(y_pred - y_true)  # Mean Bias
    se = math.sqrt(np.sum(((y_pred - y_true) - bias) ** 2) / (len(y_true) - 2))  # Standard Error

    # Return metrics as a dictionary
    return {'RMSE': rmse, 'NMSE': nmse, 'IA': ia, 'SI': si, 'NSE': nse, 'R2': r2, 'Bias': bias, 'SE': se}

# Evaluate the model on training, validation, and test sets
train_metrics = calculate_metrics(y_train, xgb_model.predict(X_train))
val_metrics = calculate_metrics(y_val, xgb_model.predict(X_val))
test_metrics = calculate_metrics(y_test, xgb_model.predict(X_test))

# Print the calculated metrics
print("Training Metrics:", train_metrics)
print("Validation Metrics:", val_metrics)
print("Test (Generalization) Metrics:", test_metrics)

## ====================== Part 5: Visualization ======================

# Function to plot measured vs predicted values with confidence intervals
def plot_measured_vs_predicted(y_true, y_pred, title, subplot_position):
    bias = np.mean(y_pred - y_true)
    se = math.sqrt(np.sum(((y_pred - y_true) - bias) ** 2) / (len(y_true) - 2))

    # Define the range for plotting
    x_range = np.linspace(-0.2, 5.5, 100)
    subplot_position.plot(y_true, y_pred, 'o', color='blue', markeredgecolor='black', markersize=8, alpha=0.7)
    subplot_position.plot(x_range, x_range + bias, color='red', linestyle='solid', label='Mean Prediction')
    subplot_position.plot(x_range, x_range + bias + 1.96 * se, color='red', linestyle='dashed', label='95% C.I. Upper')
    subplot_position.plot(x_range, x_range + bias - 1.96 * se, color='red', linestyle='dotted', label='95% C.I. Lower')

    subplot_position.set_xlim(-0.2, 5.5)
    subplot_position.set_ylim(-0.2, 5.5)
    subplot_position.set_xlabel('Measured Value ($y_{sm}/b_n$)', fontsize=14)
    subplot_position.set_ylabel('Predicted Value ($y_{sp}/b_n$)', fontsize=14)
    subplot_position.set_title(title, fontsize=16)
    subplot_position.legend(fontsize=10)
    subplot_position.grid(True, linestyle='--', alpha=0.5)

# Create plots for training, validation, and test datasets
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
plot_measured_vs_predicted(y_train, xgb_model.predict(X_train), "Training Set", axs[0])
plot_measured_vs_predicted(y_val, xgb_model.predict(X_val), "Validation Set", axs[1])
plot_measured_vs_predicted(y_test, xgb_model.predict(X_test), "Test Set (Generalization)", axs[2])

# Save and display the visualization
plt.tight_layout()
plt.savefig('Total_validation.png', dpi=600)
plt.show()


# ====================== Part 6: Applying Multiplying Factors and Calculating MAPE & Conservatism ======================

# Concatenate train, validation, and test sets
all_y_true = np.concatenate([y_train, y_val, y_test])
all_y_pred = np.concatenate([xgb_model.predict(X_train), xgb_model.predict(X_val), xgb_model.predict(X_test)])

# Define multiplying factor range
factors = np.arange(1.0, 2.4, 0.01)

# Store results
results = []

for factor in factors:
    # Apply the multiplying factor
    adjusted_predictions = all_y_pred * factor

    # Calculate MAPE
    mape = np.mean(np.abs((all_y_true - adjusted_predictions) / all_y_true)) * 100

    # Calculate Level of Conservatism (percentage of overestimated samples)
    conservatism_level = np.sum(adjusted_predictions >= all_y_true) / len(all_y_true) * 100

    # Append results
    results.append([factor, mape, conservatism_level])

# Convert results to DataFrame
df_results = pd.DataFrame(results, columns=["Factor", "MAPE", "Conservatism Level (%)"])

# Display results
print(df_results)

# ====================== Part 7: Plotting MAPE and Conservatism Level on a Single Plot ======================

plt.figure(figsize=(10, 6))

# Plot MAPE with primary y-axis
plt.plot(df_results["Factor"], df_results["MAPE"], marker='o', label="MAPE", color="blue")
plt.xlabel("Multiplying Factor")
plt.ylabel("MAPE (%)", color="blue")
plt.title("MAPE and Conservatism Level vs. Multiplying Factor")
plt.legend(loc="upper left")
plt.grid(True)

# Plot Conservatism Level with secondary y-axis
ax2 = plt.gca().twinx()
ax2.plot(df_results["Factor"], df_results["Conservatism Level (%)"], marker='o', color="orange", label="Conservatism Level")
ax2.set_ylabel("Conservatism Level (%)", color="orange")
ax2.legend(loc="upper right")

plt.show()

## ====================== Part 8: SHAP Analysis and Feature Importance Visualization ======================

# Feature names for SHAP interpretation
feature_names = [r'$y/b$', r'$b/d_{50}$', r'$V/V_c$', r'$Fr$']
scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)

# Initialize SHAP analysis
shap.initjs()
explainer = shap.Explainer(xgb_model, scaled_input_df)
shap_values = explainer(scaled_input_df)

# Save SHAP values to CSV
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_csv("/Users/rlaxo/PycharmProjects/deeplearning/Scour_field_practice1/shap_result.csv", index=False)

# Summary plot for feature importance with feature names
shap.summary_plot(shap_values.values, scaled_input_df, feature_names=feature_names)
shap.summary_plot(shap_values.values, scaled_input_df, feature_names=feature_names, plot_type="bar")

# Common background data for SHAP
feature_names = [r'$y/b$', r'$b/d_{50}$', r'$V/V_c$', r'$Fr$']
scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)

# Initialize TreeExplainer with background data
explainer = shap.TreeExplainer(xgb_model, data=scaled_input_df, feature_perturbation="interventional")

# Calculate SHAP values for all cases
shap_values_all_cases = explainer(scaled_input_df)

# Generate waterfall plots for specific cases
case_indices = [239, 400]
for idx in case_indices:
    shap.plots.waterfall(shap_values_all_cases[idx])

## ====================== Part 9: User Input for Prediction ======================

# User input for conservatism levels
conservatism_levels = [float(x) for x in input("Enter desired conservatism levels (e.g., 0.6, 0.95): ").split(",")]

# Function to automatically find multiplying factors for desired conservatism levels
def calculate_multiplying_factors(model, X, y_true, conservatism_levels):
    factors = []
    for level in conservatism_levels:
        factor = 1.0
        while True:
            adjusted_predictions = model.predict(X) * factor
            conservatism_level = np.sum(adjusted_predictions >= y_true) / len(y_true)
            if conservatism_level >= level:
                factors.append(factor)
                break
            factor += 0.01  # Gradually increase factor to achieve desired conservatism
    return factors

# Concatenate train, validation, and test data
all_X = np.concatenate([X_train, X_val, X_test])
all_y_true = np.concatenate([y_train, y_val, y_test])

# Calculate multiplying factors for each conservatism level
multiplying_factors = calculate_multiplying_factors(xgb_model, all_X, all_y_true, conservatism_levels)
print("Calculated Multiplying Factors for each Conservatism Level:", multiplying_factors)

# Input variables for prediction
b = float(input("Enter pier width (b) in ft: "))
L = float(input("Enter pier length (L) in ft: "))
theta = float(input("Enter skew angle (θ) in degrees: "))
V = float(input("Enter mean flow velocity (V) in ft/s: "))
y = float(input("Enter upstream flow depth (y) in ft: "))
d50 = float(input("Enter median particle size (d50) in mm: "))
Ku = float(input("Enter conversion constant (Ku, typically 0.3048): "))

# Step 1: Calculate V_c
if 0.1 < d50 < 1:
    uc_star = Ku * (0.037 + 0.041 * d50 ** 1.4)
elif 1 < d50 < 100:
    uc_star = Ku * (0.01 * d50 ** 0.5 - 0.0213 / d50)
Vc = 5.75 * uc_star * np.log10(5.53 * (y / 3.28) / (d50 / 1000)) * 3.28  # Convert from m/s to ft/s
Vc = np.floor(Vc * 10) / 10
print(f"Calculated V_c (rounded down to one decimal place): {Vc} ft/s")

# Step 2: Calculate b_n
b_n = b * (np.cos(np.radians(theta)) + (L / b) * np.sin(np.radians(theta)))
b_n = np.floor(b_n * 10) / 10
print(f"Calculated b_n (rounded down to one decimal place): {b_n} ft")
V_over_Vc = V / Vc
input_values = {
    r'$y/b$': [y / b_n],
    r'$b/d_{50}$': [b_n / (d50 / 1000)],
    r'$V/V_c$': [V_over_Vc],
    r'$Fr$': [V / np.sqrt(32.2 * b_n)]
}
input_df = pd.DataFrame(input_values)


def manual_normalization(input_df, scour_df):
    min_max_values = {
        r'$y/b$': (scour_df[r'$y/b$'].min(), scour_df[r'$y/b$'].max()),
        r'$b/d_{50}$': (scour_df[r'$b/d_50$'].min(), scour_df[r'$b/d_50$'].max()),
        r'$V/V_c$': (scour_df[r'$V/V_c$'].min(), scour_df[r'$V/V_c$'].max()),
        r'$Fr$': (scour_df[r'$Fr_b$'].min(), scour_df[r'$Fr_b$'].max()),
    }

    normalized_data = {}
    for feature, (min_val, max_val) in min_max_values.items():
        normalized_data[feature] = (input_df[feature] - min_val) / (max_val - min_val)

    return pd.DataFrame(normalized_data)

normalized_input_df = manual_normalization(input_df, scour)

# Generate waterfall plots for SHAP values according to conservatism levels
explainer_case_study = shap.TreeExplainer(xgb_model, data=scaled_input, feature_perturbation="interventional")
shap_values_case_study = explainer_case_study(normalized_input_df)

# Create waterfall plot for the original predictions (no conservatism)
print("\nWaterfall plot for original predictions (no conservatism):")
shap.waterfall_plot(shap.Explanation(
    values=shap_values_case_study.values[0],
    base_values=explainer_case_study.expected_value,
    data=normalized_input_df.values[0],
    feature_names=input_df.columns
))

# Create waterfall plots for each desired conservatism level
for i, factor in enumerate(multiplying_factors):
    adjusted_shap_values = shap_values_case_study.values * factor
    print(f"\nWaterfall plot for conservatism level {int(conservatism_levels[i] * 100)}%:")
    shap.waterfall_plot(shap.Explanation(
        values=adjusted_shap_values[0],
        base_values=explainer_case_study.expected_value * factor,
        data=normalized_input_df.values[0],
        feature_names=input_df.columns
    ))

# Perform prediction
predicted_dimensionless_scour_depth = xgb_model.predict(normalized_input_df)
predicted_scour_depth = predicted_dimensionless_scour_depth * b_n

print("Predicted Scour Depth (no conservatism):", predicted_scour_depth[0])
for level, factor in zip(conservatism_levels, multiplying_factors):
    conservative_scour_depth = predicted_scour_depth * factor
    print(f"Predicted Scour Depth at {int(level * 100)}% conservatism:", conservative_scour_depth[0])