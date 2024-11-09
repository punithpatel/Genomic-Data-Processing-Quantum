import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Configuration: Number of samples for each cancer type and stage
num_samples_per_type_stage = 300  # Adjust this as needed
cancer_types_by_gender = {
    "Male": ["Lung Cancer", "Colon Cancer", "Prostate Cancer"],
    "Female": ["Lung Cancer", "Colon Cancer", "Breast Cancer", "Ovarian Cancer"]
}
stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]

# Define Feature Ranges with Noise and Overlap
def add_gaussian_noise(low, high, noise_scale=0.3):
    noisy_low = round(np.random.normal(low, noise_scale), 2)
    noisy_high = round(np.random.normal(high, noise_scale), 2)
    return (min(noisy_low, noisy_high), max(noisy_low, noisy_high))

feature_ranges = {
    "Tumor_Size": {
        "Stage I": add_gaussian_noise(0.5, 2.2),
        "Stage II": add_gaussian_noise(1.8, 5.2),
        "Stage III": add_gaussian_noise(4.8, 7.2),
        "Stage IV": add_gaussian_noise(6.8, 10.5)
    },
    "Lymph_Nodes": {
        "Stage I": round(random.uniform(0.05, 0.15), 2),
        "Stage II": round(random.uniform(0.35, 0.45), 2),
        "Stage III": round(random.uniform(0.65, 0.75), 2),
        "Stage IV": round(random.uniform(0.85, 0.95), 2)
    },
    "Metastasis": {
        "Stage I": round(random.uniform(0.01, 0.05), 2),
        "Stage II": round(random.uniform(0.08, 0.12), 2),
        "Stage III": round(random.uniform(0.25, 0.35), 2),
        "Stage IV": round(random.uniform(0.7, 0.8), 2)
    }
}

# Define Mutation Probabilities with Added Noise
mutation_probabilities = {
    "TP53_Mutation": {
        "Lung Cancer": {stage: round(random.uniform(0.15 + 0.1*i, 0.25 + 0.1*i), 2) for i, stage in enumerate(stages)},
        "Colon Cancer": {stage: round(random.uniform(0.1 + 0.1*i, 0.2 + 0.1*i), 2) for i, stage in enumerate(stages)},
        "Breast Cancer": {stage: round(random.uniform(0.05 + 0.1*i, 0.15 + 0.1*i), 2) for i, stage in enumerate(stages)},
        "Ovarian Cancer": {stage: round(random.uniform(0.05 + 0.1*i, 0.15 + 0.1*i), 2) for i, stage in enumerate(stages)},
        "Prostate Cancer": {stage: round(random.uniform(0.03 + 0.05*i, 0.07 + 0.05*i), 2) for i, stage in enumerate(stages)}
    },
    "BRCA1_Mutation": {
        "Lung Cancer": {stage: 0.0 for stage in stages},
        "Colon Cancer": {stage: 0.0 for stage in stages},
        "Breast Cancer": {stage: round(random.uniform(0.15 + 0.1*i, 0.25 + 0.1*i), 2) for i, stage in enumerate(stages)},
        "Ovarian Cancer": {stage: round(random.uniform(0.2 + 0.1*i, 0.3 + 0.1*i), 2) for i, stage in enumerate(stages)},
        "Prostate Cancer": {stage: 0.0 for stage in stages}
    }
}



# Updated gene expression ranges with non-overlapping values to reduce overfitting
gene_expression_ranges = {
    "Lung Cancer": {
        "EGFR": (0.95, 1.15),
        "KRAS": (1.2, 1.4),
        "TP53": (1.35, 1.55),
        "ALK": (1.25, 1.45),
        "BRAF": (1.7, 1.9),
        "RET": (1.85, 2.05)
    },
    "Colon Cancer": {
        "APC": (0.5, 0.7),
        "KRAS": (0.75, 0.95),
        "TP53": (0.9, 1.1),
        "PIK3CA": (1.05, 1.25),
        "SMAD4": (1.3, 1.5),
        "NRAS": (1.45, 1.65),
        "PTEN": (1.6, 1.8),
        "TGFBR2": (1.75, 1.95)
    },
    "Breast Cancer": {
        "BRCA1": (0.55, 0.75),
        "BRCA2": (0.7, 0.9),
        "TP53": (0.85, 1.05),
        "HER2": (1.0, 1.2),
        "PIK3CA": (1.25, 1.45),
        "GATA3": (1.5, 1.7),
        "ESR1": (1.65, 1.85),
        "CDH1": (1.8, 2.0)
    },
    "Ovarian Cancer": {
        "BRCA1": (0.4, 0.6),
        "BRCA2": (0.6, 0.8),
        "TP53": (0.75, 0.95),
        "PIK3CA": (1.0, 1.2),
        "RB1": (1.2, 1.4)
    },
    "Prostate Cancer": {
        "PTEN": (0.65, 0.85),
        "TP53": (0.9, 1.1),
        "BRCA2": (1.1, 1.3),
        "FOXA1": (1.3, 1.5),
        "Androgen Receptor": (1.4, 1.6)
    }
}




# Define age range by cancer type
age_range_by_cancer_type = {
    "Lung Cancer": (30, 80),
    "Colon Cancer": (30, 80),
    "Breast Cancer": (20, 80),
    "Ovarian Cancer": (20, 80),
    "Prostate Cancer": (40, 80)
}

# Helper function to get expression for associated or other types
def get_expression(gene, cancer_type):
    if gene in gene_expression_ranges[cancer_type]:
        return np.random.uniform(*gene_expression_ranges[cancer_type][gene])
    else:
        return np.random.uniform(0.1, 0.3)  # Background level for non-associated cancer types

# Generate synthetic dataset
data = []
sample_id = 1
for stage in stages:
    for gender in ["Male", "Female"]:
        possible_cancer_types = cancer_types_by_gender[gender]
        for cancer_type in possible_cancer_types:
            for _ in range(num_samples_per_type_stage):
                sample = {
                    "Sample_ID": f"S{sample_id:04d}",
                    "Age": np.random.randint(*age_range_by_cancer_type[cancer_type]),
                    "Gender": gender,
                    "Tumor_Size": np.random.uniform(*feature_ranges["Tumor_Size"][stage]),
                    "Lymph_Nodes": "Yes" if np.random.rand() < feature_ranges["Lymph_Nodes"][stage] else "No",
                    "Metastasis": "Yes" if np.random.rand() < feature_ranges["Metastasis"][stage] else "No"                   
                }
                # Add mutations
                sample["TP53_Mutation"] = "Yes" if np.random.rand() < mutation_probabilities["TP53_Mutation"][cancer_type][stage] else "No"
                sample["BRCA1_Mutation"] = "Yes" if np.random.rand() < mutation_probabilities["BRCA1_Mutation"][cancer_type][stage] else "No"

                # Gene expressions for the specific cancer type
                for gene in gene_expression_ranges[cancer_type].keys():
                    sample[gene] = get_expression(gene, cancer_type)

                # Background noise for other cancer types
                for other_type in cancer_types_by_gender["Male"] + cancer_types_by_gender["Female"]:
                    if other_type != cancer_type:
                        for gene in gene_expression_ranges[other_type].keys():
                            if gene not in sample:
                                sample[gene] = get_expression(gene, other_type)

                # Add general gene expressions for background noise
                for i in range(1, 11):
                    sample[f"Gene{i}_Expr"] = np.random.uniform(0.5, 1.0)
                
                sample["Cancer_Type"] = cancer_type
                sample["Cancer_Stage"] = stage
                data.append(sample)
                sample_id += 1

# Convert to DataFrame and reorder columns
df = pd.DataFrame(data)
column_order = ["Sample_ID", "Age", "Gender"] + \
               [col for col in df.columns if col not in ["Sample_ID", "Age", "Gender", "Cancer_Type", "Cancer_Stage"]] + \
               ["Cancer_Type", "Cancer_Stage"]
df = df[column_order]

# Save to CSV
file_path = "synthetic_cancer_dataset.csv"
df.to_csv(file_path, index=False)
print(f"Dataset saved to: {file_path}")




# import pandas as pd

# # Load the dataset
# file_path = 'synthetic_cancer_dataset.csv'
# df = pd.read_csv(file_path)

# # Separate Sample_ID column
# sample_ids = df['Sample_ID']
# data = df.drop(columns=['Sample_ID'])

# # Shuffle the data
# data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# # Reattach the Sample_ID column in its original order
# df_shuffled = pd.concat([sample_ids, data_shuffled], axis=1)

# # Save the shuffled dataset
# df_shuffled.to_csv('raw_dataset.csv', index=False)

# print("Dataset shuffled and saved as 'raw_dataset.csv'")


