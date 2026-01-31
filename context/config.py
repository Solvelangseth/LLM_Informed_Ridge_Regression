"""Centralized configuration for experiments."""

import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Hyperparameter grids
C_GRID = np.logspace(-4, 4, 41)  # For L2 logistic regression
ALPHA_GRID = np.logspace(-4, 4, 41)  # For target-informed models
GAMMA_GRID = np.linspace(0, 1, 101)  # For mixed models

# Numerical stability
EPS = 1e-12

# Train/validation/test splits
TEST_SIZE = 0.20  # Hold-out test set
VAL_SIZE = 0.25   # Validation from remaining (0.8 * 0.25 = 0.2 overall)

# Model settings
MAX_ITER = 5000  # For logistic regression
FIT_INTERCEPT = True

# Evaluation threshold
CLASSIFICATION_THRESHOLD = 0.5
