# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

def topsis(decision_matrix, criteria_weights):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    Implementation for multi-criteria decision analysis
    
    Parameters:
    decision_matrix : pd.DataFrame - M x N matrix with M alternatives and N criteria
    criteria_weights : np.array - Weights for each criterion (sum to 1)
    
    Returns:
    np.array - TOPSIS scores for each alternative
    """
    # Normalize decision matrix using min-max scaling
    matrix_min = np.min(decision_matrix)
    matrix_max = np.max(decision_matrix)
    normalized_matrix = (decision_matrix - matrix_min) / (matrix_max - matrix_min)

    # Create weighted normalized matrix
    weighted_matrix = normalized_matrix * criteria_weights

    # Determine ideal solutions
    ideal_positive = np.max(weighted_matrix, axis=0)
    ideal_negative = np.min(weighted_matrix, axis=0)
    
    # Invert cost criteria (columns 1,4,5,9 when 0-indexed)
    cost_indices = [1, 4, 5, 9]
    ideal_positive[cost_indices], ideal_negative[cost_indices] = (
        ideal_negative[cost_indices], 
        ideal_positive[cost_indices]
    )

    # Calculate Euclidean distances
    dist_positive = np.sqrt(np.sum((weighted_matrix - ideal_positive) ** 2, axis=1))
    dist_negative = np.sqrt(np.sum((weighted_matrix - ideal_negative) ** 2, axis=1))

    # Calculate relative closeness scores
    closeness_scores = dist_negative / (dist_positive + dist_negative + 1e-9)  # Avoid division by zero

    return closeness_scores

# Monte Carlo simulation parameters
NUM_PRIMARY_CRITERIA = 4
NUM_SIMULATIONS = 50
MIN_WEIGHT = 0.1
MAX_WEIGHT = 0.5

# Generate valid weight combinations using MCS
simulated_weights = []
while len(simulated_weights) < NUM_SIMULATIONS:
    raw_weights = np.random.uniform(low=MIN_WEIGHT, high=MAX_WEIGHT, size=NUM_PRIMARY_CRITERIA)
    
    if np.all(raw_weights <= MAX_WEIGHT) and np.all(raw_weights >= MIN_WEIGHT):
        normalized_weights = raw_weights / raw_weights.sum()
        simulated_weights.append(normalized_weights)

simulated_weights = np.array(simulated_weights)

# Visualize weight distributions
plt.figure(figsize=(10, 6))
for i in range(NUM_PRIMARY_CRITERIA):
    plt.hist(simulated_weights[:, i], bins=20, alpha=0.5, label=f'Criterion {i+1}')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Primary Criteria Weight Distributions')
plt.legend()
plt.savefig('weight_distributions.png', dpi=300)
plt.close()

# Allocate weights to secondary criteria
SECONDARY_CRITERIA_MAP = [4, 1, 2, 3]  # Number of sub-criteria per primary criterion
detailed_weights = np.zeros((NUM_SIMULATIONS, 10))

for sim_idx in range(NUM_SIMULATIONS):
    current_idx = 0
    for primary_idx, num_secondary in enumerate(SECONDARY_CRITERIA_MAP):
        end_idx = current_idx + num_secondary
        detailed_weights[sim_idx, current_idx:end_idx] = (
            simulated_weights[sim_idx, primary_idx] / num_secondary
        )
        current_idx = end_idx

# Load decision matrix from Excel
decision_data = pd.read_excel('data_sample.xlsx', usecols="B:K")

# Run TOPSIS simulations
simulation_results = []
for weight_vector in detailed_weights:
    scores = topsis(decision_data.values, weight_vector)
    simulation_results.append({
        'weights': weight_vector,
        'scores': scores
    })

# Create output DataFrames
weights_df = pd.DataFrame([res['weights'] for res in simulation_results],
                         columns=[f'Criterion_{i+1}' for i in range(10)])
scores_df = pd.DataFrame([res['scores'] for res in simulation_results]).T

# Generate rankings
rankings_df = scores_df.rank(axis=0, ascending=False, method='min')

# Excel export configuration
def export_to_excel(dataframe, filename, sheet_size=10):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        num_sheets = int(np.ceil(dataframe.shape[1] / sheet_size))
        for sheet_num in range(num_sheets):
            start_col = sheet_num * sheet_size
            end_col = start_col + sheet_size
            sheet_data = dataframe.iloc[:, start_col:end_col]
            sheet_data.to_excel(writer, sheet_name=f'Sim_{start_col+1}-{end_col}')

# Export results
weights_df.to_excel('simulation_weights.xlsx', index=False)
export_to_excel(scores_df, 'topsis_scores.xlsx')
export_to_excel(rankings_df, 'topsis_rankings.xlsx')

print("Simulation completed successfully. Results exported to Excel files.")