import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the shuffled dataset
file_path = 'raw_dataset.csv'
df = pd.read_csv(file_path)

# Identify columns for preprocessing
categorical_columns = ['Gender', 'Cancer_Type', 'Cancer_Stage', 'TP53_Mutation', 'BRCA1_Mutation']
numerical_columns = [col for col in df.columns if col not in categorical_columns + ['Sample_ID']]

# Step 1: Encode Categorical Variables
# Convert binary columns (Yes/No) to numeric (0/1)
df['Lymph_Nodes'] = df['Lymph_Nodes'].map({'Yes': 1, 'No': 0})
df['Metastasis'] = df['Metastasis'].map({'Yes': 1, 'No': 0})

# Using Label Encoding for binary columns and OneHotEncoding for multi-category columns
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['TP53_Mutation'] = LabelEncoder().fit_transform(df['TP53_Mutation'])
df['BRCA1_Mutation'] = LabelEncoder().fit_transform(df['BRCA1_Mutation'])

# One-hot encode Cancer_Type and Cancer_Stage
df = pd.get_dummies(df, columns=['Cancer_Type', 'Cancer_Stage'], drop_first=True)

# Step 2: Scale Numerical Features
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 3: Separate features and target variables
X = df.drop(columns=['Sample_ID'])  # Drop Sample_ID if it's not needed
y_stage = df[[col for col in df.columns if 'Cancer_Stage_' in col]]
y_type = df[[col for col in df.columns if 'Cancer_Type_' in col]]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_stage_train, y_stage_test, y_type_train, y_type_test = train_test_split(
    X, y_stage, y_type, test_size=0.2, random_state=42
)

# Confirm shapes of split data
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Optional: Save preprocessed data for training
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_stage_train.to_csv('y_stage_train.csv', index=False)
y_stage_test.to_csv('y_stage_test.csv', index=False)
y_type_train.to_csv('y_type_train.csv', index=False)
y_type_test.to_csv('y_type_test.csv', index=False)

print("Preprocessed data saved for training and testing.")
