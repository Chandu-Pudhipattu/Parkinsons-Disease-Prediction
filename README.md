**Parkinson's Disease Prediction using XGBoost**

**Project Overview**

This project aims to predict Parkinson's disease using machine learning techniques, specifically the XGBoost algorithm. The dataset used contains biomedical voice measurements from patients, which are classified into healthy individuals and those diagnosed with Parkinson's disease. The goal is to develop a model that accurately distinguishes between these two categories.

**Dataset**

The dataset contains 24 features, including:

Categorical Features:

name: Subject names (transformed into numerical identifiers).

Numerical Features:

MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz) (Frequency-related measures).

MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP (Jitter metrics).

MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA (Shimmer metrics).

NHR, HNR (Noise-to-harmonics ratio).

RPDE, DFA (Nonlinear dynamical complexity measures).

spread1, spread2, D2, PPE (Various vocal measurements).

status (Target variable: 1 for Parkinson’s disease, 0 for healthy).

**Workflow**

1. **Data Preprocessing**

Data Cleaning:

Removed irrelevant parts of the name column (e.g., prefixes and suffixes).

Encoded the name column into numerical subject identifiers.

Feature Scaling:

Normalized numerical features for better model performance.

Splitting Data:

Divided the dataset into training and test sets.

2. **Exploratory Data Analysis (EDA)**

Visualized feature distributions and correlations:

Correlation Heatmap: Identified features strongly correlated with the target.

Heatmap: Identified the missing values

Histplot: Identified the distributions between features

3. **Model Implementation**


**Algorithm**: XGBoost Classifier

Tuned hyperparameters using GridSearchCV to optimize model performance.

Best Parameters:
```bash
learning_rate: 0.1

max_depth: 5

n_estimators: 100
```
4. Model Evaluation

Metrics Used:
```bash
Accuracy

Precision, Recall, and F1-score

Confusion Matrix

Cross-Validation:
```
Performed 5-fold cross-validation to ensure the model generalizes well.

Best cross-validation score: 95.56%.

### Results

Training Accuracy: Achieved high training accuracy.

Test Accuracy: Evaluated the model on the test set, achieving 98.31% accuracy.

1. Confusion matrix:
   ```bash
   [[14  1]
   [ 0 44]]
   ```

3. Install dependencies:
   ```bash
                precision    recall  f1-score   support

          0       1.00      0.93      0.97        15
          1       0.98      1.00      0.99        44

   accuracy                           0.98        59
   macro avg       0.99      0.97      0.98        59
   ```

## Future Work
- **Feature Engineering:** Explore additional feature extraction techniques.
- **Advanced Models:** Experiment with ensemble models and deep learning.
- **Deployment:** Build a web application to provide real-time predictions using Django or Flask.
- **Explainability:** Implement SHAP or LIME for model interpretability.
   ```




