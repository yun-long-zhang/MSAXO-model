import pandas as pd
from pulp import *
import time
import os
import csv
import gc

# Load source-sink matching data
data = pd.read_csv(os.path.join('data2', 'input_data.csv'))  # Original filename kept as per data

# Generate valid source-sink combinations from data
valid_combinations = [(row.i, row.j) for row in data.itertuples()]

# Initialize optimization problem
prob = LpProblem("CO2_Transport_Optimization", LpMinimize)

# Create decision variables (X_ij: CO2 flow from source i to sink j)
X = LpVariable.dicts("Transport", valid_combinations, lowBound=0, cat=LpContinuous)

# Precompute parameters for optimization
costs = data.set_index(['i', 'j'])['CCS_cost'].to_dict()
emissions = data.groupby('i')['emissions'].first().to_dict()
storage = (data.groupby('j')['storage'].first() / 20).to_dict()  # Annual storage capacity
source_ranks = data.set_index(['i', 'j'])['source_rank'].to_dict()
storage_ranks = data.set_index(['i', 'j'])['storage_rank'].to_dict()

# Set objective function: Minimize total transportation cost
prob += lpSum(X[i, j] * costs.get((i, j), 0) for i, j in valid_combinations)

# Parameter sensitivity analysis loop
for storage_factor_value in range(10, 101, 5):  # 10% to 100% in 5% increments
    for source_factor_value in range(10, 101, 5):
        storage_factor = storage_factor_value / 100  # Convert to decimal
        source_factor = source_factor_value / 100

        # Add emission constraints for each source
        for i in data['i'].unique():
            prob += lpSum(X[i, j] for j in data.loc[data['i'] == i, 'j'].unique()) <= emissions[i]
        
        # Add storage constraints for each sink
        for j in data['j'].unique():
            prob += lpSum(X[i, j] for i in data.loc[data['j'] == j, 'i'].unique()) <= storage[j]

        # Add priority ranking constraints
        for i, j in valid_combinations:
            # Source quality constraint (1040 = max source rank * max factor)
            prob += X[i, j] * source_ranks[i, j] <= 1040 * source_factor * X[i, j]
            
            # Sink priority constraint (466 = max storage rank * max factor)
            prob += X[i, j] * storage_ranks[i, j] <= 466 * storage_factor * X[i, j]

        # Minimum total transportation requirement
        prob += lpSum(X[i, j] for i, j in valid_combinations) >= 700

        # Solve optimization problem
        start_time = time.perf_counter()
        prob.solve()
        solve_duration = time.perf_counter() - start_time
        
        print(f"Optimization time: {solve_duration:.2f} s")
        print(f"Current parameters - Storage factor: {storage_factor:.2f}, Source factor: {source_factor:.2f}")

        # Handle optimization results
        if prob.status != LpStatusOptimal:
            # Log failed optimization attempts
            with open('objectives.csv', 'a') as f:
                csv.writer(f).writerow([storage_factor, source_factor, None])
        else:
            # Record successful optimization results
            total_cost = value(prob.objective)
            with open('objectives.csv', 'a') as f:
                csv.writer(f).writerow([storage_factor, source_factor, total_cost])
            
            # Generate detailed results dataframe
            results = pd.DataFrame(columns=[
                'i', 'j', 'Xij', 'Xi_longitude', 'Xi_latitude',
                'Xj_longitude', 'Xj_latitude', 'emissions', 'storage',
                'source_rank', 'storage_rank', 'cost', 'total_cost'
            ])
            
            for i, j in valid_combinations:
                if X[i, j].value() > 1e-6:  # Ignore negligible flows
                    row = data.loc[(data['i'] == i) & (data['j'] == j)].iloc[0]
                    record = {
                        'i': i, 'j': j, 'Xij': X[i, j].value(),
                        'Xi_longitude': row.Xi, 'Xi_latitude': row.Yi,
                        'Xj_longitude': row.Xj, 'Xj_latitude': row.Yj,
                        'emissions': emissions[i], 'storage': storage[j],
                        'source_rank': source_ranks[(i, j)],
                        'storage_rank': storage_ranks[(i, j)],
                        'cost': costs[(i, j)],
                        'total_cost': X[i, j].value() * costs[(i, j)]
                    }
                    results = results.append(record, ignore_index=True)
            
            # Save parameter-specific results
            results.to_csv(f'result/results_{storage_factor:.2f}-{source_factor:.2f}.csv', index=False)

        # Reset problem for next iteration
        prob.constraints.clear()
        gc.collect()