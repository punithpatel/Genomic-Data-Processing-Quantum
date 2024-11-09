# Import necessary libraries and modules
import random  # Standard library for setting random seed
import pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import RealAmplitudes
from qiskit_ibm_provider import IBMProvider  # Corrected import for IBMProvider
from qiskit_aer import AerSimulator

API_TOKEN='121532';
# Initialize IBMProvider with your API token
IBMProvider.save_account(API_TOKEN, overwrite=True)  # Replace with your actual IBM API token
provider = IBMProvider.load_account()

class ScipyOptimizer:
    def __init__(self, maxiter=100):
        self.maxiter = maxiter

    def optimize(self, objective_func, initial_params):
        # Use scipy's minimize function to optimize
        result = minimize(objective_func, initial_params, method='L-BFGS-B', options={'maxiter': self.maxiter})
        return result.x, result.fun, result.nit

class QuantumCancerPredictionModel:
    def __init__(self, stage_num_classes=3, type_num_classes=4):
        # Set a random seed for reproducibility
        random.seed(42)

        # Cancer stage quantum model setup with all features
        stage_feature_dim = 5  # Using all features for cancer stage
        stage_quantum_circuit = RealAmplitudes(num_qubits=stage_feature_dim, reps=2)
        stage_qnn = EstimatorQNN(feature_map=stage_quantum_circuit, quantum_instance=AerSimulator())
        stage_optimizer = ScipyOptimizer(maxiter=100)
        self.vqc_stage = VQC(num_classes=stage_num_classes, quantum_neural_network=stage_qnn, 
                             optimizer=stage_optimizer)

        # Cancer type quantum model setup with all features
        type_feature_dim = 30  # Using all features for cancer type
        type_quantum_circuit = RealAmplitudes(num_qubits=type_feature_dim, reps=2)
        type_qnn = EstimatorQNN(feature_map=type_quantum_circuit, quantum_instance=AerSimulator())
        type_optimizer = ScipyOptimizer(maxiter=100)
        self.vqc_type = VQC(num_classes=type_num_classes, quantum_neural_network=type_qnn, 
                            optimizer=type_optimizer)

    def train_stage_model(self, X_train, y_train):
        # Train the quantum classifier for cancer stage
        self.vqc_stage.fit(X_train, y_train)
    
    def predict_stage(self, X):
        # Predict cancer stage
        return self.vqc_stage.predict(X)

    def train_type_model(self, X_train, y_train):
        # Train the quantum classifier for cancer type
        self.vqc_type.fit(X_train, y_train)

    def predict_type(self, X):
        # Predict cancer type
        return self.vqc_type.predict(X)

# Function to load datasets
def load_data():
    # Load preprocessed training data for cancer stage and type
    X_train = pd.read_csv("X_train_preprocessed.csv")
    y_stage_train = pd.read_csv("y_stage_train.csv").values.ravel()  # Ensure y is a 1D array
    y_type_train = pd.read_csv("y_type_train.csv").values.ravel()

    # Load preprocessed test data
    X_test = pd.read_csv("X_test_preprocessed.csv")
    
    return X_train, y_stage_train, y_type_train, X_test

# Usage Example
if __name__ == "__main__":
    # Load data
    X_train, y_stage_train, y_type_train, X_test = load_data()

    # Initialize the quantum cancer prediction model
    qc_model = QuantumCancerPredictionModel(stage_num_classes=3, type_num_classes=4)

    # Train both models
    qc_model.train_stage_model(X_train, y_stage_train)
    qc_model.train_type_model(X_train, y_type_train)

    # Predict on test data for stage and type
    stage_prediction = qc_model.predict_stage(X_test)
    type_prediction = qc_model.predict_type(X_test)

    print("Predicted cancer stage:", stage_prediction)
    print("Predicted cancer type:", type_prediction)

