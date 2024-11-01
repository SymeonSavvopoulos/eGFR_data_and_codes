import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset with 3 datasets, 15 features, 100 samples
# Features characteristics based on the table provided by the user
features = [
    ('c', 2), ('c', 2), ('c', 2), ('c', 2), 
    ('c', 2)
]


# import pandas as pd

# # Load the CSV file into a DataFrame
dataset1 = pd.read_csv('D:/Dropbox/eGFR_article_data_algorithms-main/Tables_1_2/organ_train_no_rej.csv')
dataset2 = pd.read_csv('D:/Dropbox/eGFR_article_data_algorithms-main/Tables_1_2/organ_train_one_rej.csv')
dataset3 = pd.read_csv('D:/Dropbox/eGFR_article_data_algorithms-main/Tables_1_2/organ_train_mult_rej.csv')

# # # Display the DataFrame
# # print(df.head())  # This shows the first few rows of the DataFrame

# # Generating the datasets with 100 samples each
# dataset1 = pd.DataFrame()
# dataset2 = pd.DataFrame()
# dataset3 = pd.DataFrame()

# Populating the datasets based on the feature characteristics
# for idx, feature in enumerate(features):
#     if feature[0] == 'n':
#         # Numerical feature
#         dataset1[f'Feature_{idx+1}'] = np.random.normal(loc=10, scale=2, size=100)
#         dataset2[f'Feature_{idx+1}'] = np.random.normal(loc=12, scale=2.5, size=100)
#         dataset3[f'Feature_{idx+1}'] = np.random.normal(loc=9, scale=1.5, size=100)
#     elif feature[0] == 'c':
#         # Categorical feature with given number of categories
#         dataset1[f'Feature_{idx+1}'] = np.random.choice(range(1, feature[1] + 1), size=100)
#         dataset2[f'Feature_{idx+1}'] = np.random.choice(range(1, feature[1] + 1), size=100)
#         dataset3[f'Feature_{idx+1}'] = np.random.choice(range(1, feature[1] + 1), size=100)

# Creating the summary table
summary_table = []

for idx, feature in enumerate(features):
    if feature[0] == 'n':
        # Numerical: Calculate mean, standard deviation, and Kruskal-Wallis p-value
        mean = [dataset1[f'Feature_{idx+1}'].mean(), dataset2[f'Feature_{idx+1}'].mean(), dataset3[f'Feature_{idx+1}'].mean()]
        std_dev = [dataset1[f'Feature_{idx+1}'].std(), dataset2[f'Feature_{idx+1}'].std(), dataset3[f'Feature_{idx+1}'].std()]
        _, p_value = kruskal(dataset1[f'Feature_{idx+1}'], dataset2[f'Feature_{idx+1}'], dataset3[f'Feature_{idx+1}'])
        summary_table.append([f'Feature_{idx+1}', np.mean(mean), np.mean(std_dev), "-", p_value])
    elif feature[0] == 'c':
        # Categorical: Calculate category counts, proportions, and chi-square p-value
        category_counts = pd.concat([
            dataset1[f'Feature_{idx+1}'].value_counts(normalize=False),
            dataset2[f'Feature_{idx+1}'].value_counts(normalize=False),
            dataset3[f'Feature_{idx+1}'].value_counts(normalize=False)
        ], axis=1).fillna(0).astype(int)

        # Chi-square test
        _, p_value, _, _ = chi2_contingency(category_counts.T)
        
        # Format the category counts and proportions
        categories = category_counts.mean(axis=1).to_dict()
        category_summary = ', '.join([f'{cat}: {int(count)} ({count/100:.2%})' for cat, count in categories.items()])
        
        summary_table.append([f'Feature_{idx+1}', "-", "-", category_summary, p_value])

# Convert to DataFrame for display
summary_df = pd.DataFrame(summary_table, columns=["Feature", "Average", "Std Dev", "Category Counts (Proportion)", "p-value"])

# Display the table
#import ace_tools as tools; tools.display_dataframe_to_user(name="Feature Summary Table", dataframe=summary_df)

# Updating the table to include averages and SDs for numerical features for all three datasets, and category counts for each dataset

summary_table_detailed = []

for idx, feature in enumerate(features):
    if feature[0] == 'n':
        # Numerical: Calculate mean ± SD for each dataset
        avg_sd_1 = f"{dataset1[f'Feature_{idx+1}'].mean():.2f} ± {dataset1[f'Feature_{idx+1}'].std():.2f}"
        avg_sd_2 = f"{dataset2[f'Feature_{idx+1}'].mean():.2f} ± {dataset2[f'Feature_{idx+1}'].std():.2f}"
        avg_sd_3 = f"{dataset3[f'Feature_{idx+1}'].mean():.2f} ± {dataset3[f'Feature_{idx+1}'].std():.2f}"
        _, p_value = kruskal(dataset1[f'Feature_{idx+1}'], dataset2[f'Feature_{idx+1}'], dataset3[f'Feature_{idx+1}'])
        summary_table_detailed.append([f'Feature_{idx+1}', avg_sd_1, avg_sd_2, avg_sd_3, "-", p_value])
    elif feature[0] == 'c':
        # Categorical: Calculate category counts and proportions for each dataset
        category_counts_1 = dataset1[f'Feature_{idx+1}'].value_counts(normalize=False)
        category_counts_2 = dataset2[f'Feature_{idx+1}'].value_counts(normalize=False)
        category_counts_3 = dataset3[f'Feature_{idx+1}'].value_counts(normalize=False)

        # Combine category counts for Chi-square test
        combined_category_counts = pd.concat([category_counts_1, category_counts_2, category_counts_3], axis=1).fillna(0).astype(int)
        _, p_value, _, _ = chi2_contingency(combined_category_counts.T)
        
        # Format category counts and proportions for each dataset
        cat_1_summary = ', '.join([f'{cat}: {count} ({count/len(dataset1):.2%})' for cat, count in category_counts_1.items()])
        cat_2_summary = ', '.join([f'{cat}: {count} ({count/len(dataset2):.2%})' for cat, count in category_counts_2.items()])
        cat_3_summary = ', '.join([f'{cat}: {count} ({count/len(dataset3):.2%})' for cat, count in category_counts_3.items()])
        
        summary_table_detailed.append([f'Feature_{idx+1}', cat_1_summary, cat_2_summary, cat_3_summary, "-", p_value])

# Convert to DataFrame for display
summary_df_detailed = pd.DataFrame(summary_table_detailed, columns=["Feature", "Dataset 1 (Mean ± SD / Category Proportion)", 
                                                                    "Dataset 2 (Mean ± SD / Category Proportion)", 
                                                                    "Dataset 3 (Mean ± SD / Category Proportion)", 
                                                                    "Category Counts (Proportion)", 
                                                                    "p-value"])

# Display the table
#tools.display_dataframe_to_user(name="Detailed Feature Summary Table", dataframe=summary_df_detailed)
