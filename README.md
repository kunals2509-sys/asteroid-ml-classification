Classifying Potentially Hazardous Asteroids from NASA JPL Orbital Data
Machine Learning Term Project — Group 16

Problem Statement
NASA's Jet Propulsion Laboratory designates a subset of catalogued asteroids as
Potentially Hazardous Asteroids (PHAs) — fewer than 1% of objects in the Small-Body
Database. This project investigates whether the PHA designation can be predicted from
orbital and physical parameters using supervised machine learning, and whether that
predictive signal survives once the parameters that define the PHA criterion
(minimum orbit intersection distance and absolute magnitude) are withheld.
Dataset
Source: NASA JPL Small-Body Database asteroid catalogue, via
Kaggle
Setup: place the CSV at the path set by `DATA_PATH` in the config cell
(defaults to `dataset.csv` in the working directory)
Target: `pha` (Y/N), encoded as binary (1 = hazardous)
Environment
Python 3 with:
`numpy`, `pandas`, `matplotlib`
`scikit-learn`
`imbalanced-learn` (for the SMOTE comparison in Section 7.2)
Key configuration variables (top of the notebook):
`RANDOM_STATE = 42` — fixed seed for reproducibility
`SAMPLE_ROWS = 150_000` — stratified subsample size used for the full workflow
(set to `None` to use the complete catalogue)
`TUNE_ROWS = 30_000` — stratified slice used for cross-validation and hyperparameter search
Methodology
Data cleaning — drop identifier/metadata columns and duplicate date fields, resolve
and encode the target, drop unlabeled records, assess field completeness.
EDA — class distribution and imbalance, distribution of key orbital parameters by
class, hazard prevalence by dynamical class, collinearity among orbital elements.
Preprocessing — stratified 80/20 train/test split; features grouped into plain
numeric, skewed numeric (signed-log + standardize), physical measurements (impute +
missingness indicator), and categorical (one-hot); all steps wrapped in scikit-learn
`Pipeline`/`ColumnTransformer` objects fit only on training data.
Modeling — three classifiers compared: Logistic Regression, Random Forest, and
Histogram-based Gradient Boosting, with class imbalance handled via balanced class
weighting (SMOTE evaluated as an alternative). Hyperparameters tuned with
`RandomizedSearchCV` under stratified 5-fold cross-validation, scored on average
precision (PR AUC).
Three experiments, evaluated once on the held-out test set:
A — Complete feature set
B — Excluding `moid`, `moid_ld`, `H` (the parameters that define the PHA rule)
C — Orbital elements only (B, plus removing `orbit_class` and `neo`, which encode
Earth-proximity indirectly)
Key Results
Experiment	Best model	PR AUC (average precision)
A: all features	Histogram Gradient Boosting	0.9949
B: excluding moid & H	Random Forest	0.3846
C: orbital elements only	Random Forest	0.4073
Near-perfect performance in Experiment A is definitional: the models recover the JPL
threshold rule on `moid` and `H` rather than learning a genuinely empirical relationship.
Performance drops substantially once those fields are withheld, but a real signal remains
in the raw orbital elements (semi-major axis, eccentricity, perihelion distance, etc.).
Ensemble methods (Random Forest, Hist Gradient Boosting) retain far more performance than
Logistic Regression as features are removed, indicating the residual relationship is
non-linear.
Histogram Gradient Boosting is the recommended model overall — best on the full
feature set and competitive with Random Forest on the reduced sets, at a fraction of the
training time.
See Section 12 of the notebook for full limitations (observational selection bias in the
catalogue, sparse physical measurements, small positive test-set count, single train/test
split, threshold left at default 0.5, etc.) and suggested further work (recall-targeted
threshold selection for planetary-defense use cases).
Notebook Structure
Introduction / problem statement
Environment and configuration
Data acquisition and quality assessment
Exploratory data analysis
Methodology overview
Outlier analysis, partitioning and preprocessing
Model implementation, class imbalance and hyperparameter tuning
Experiment A — evaluation on the complete feature set
Feature importance and Experiment B
Experiment C — orbital elements in isolation
Comparative results
Conclusions and limitations
Contribution summary
How to Run
Download the dataset CSV from Kaggle (link above) and place it at the path set by
`DATA_PATH`, or update `DATA_PATH` to point to your copy.
Install dependencies:
```bash
   pip install numpy pandas matplotlib scikit-learn imbalanced-learn
   ```
Run all cells top to bottom in Jupyter (each data-processing cell assigns a new
dataframe name rather than reassigning, so the notebook is largely idempotent under
re-execution).
Contributions
Author	Sections	Focus
Leonardo Rodrigues Fabbro	3, 12	Data loading/schema checks, data dictionary, cleaning, target encoding, duplicate/completeness checks, subsampling, conclusions
Madhurishitha Boddu	4, 8	EDA (class distribution, feature distributions, dynamical class, correlation), evaluation function, test-set metrics, confusion matrices, PR curves, comparative results
Shunmuka Valsa	6, 10	Outlier analysis, partitioning, feature grouping, signed-log transform, imputation/missingness indicators, one-hot encoding, ColumnTransformer, Experiment C
Kunal Sharma	7, 9	Model pipelines, class weighting, SMOTE evaluation, cross-validation, hyperparameter search, refitting, feature importance, Experiment B
All members contributed to the proposal, attended project meetings, and participated in the
presentation.
