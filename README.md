# House Price Prediction — Linear Regression to Model Comparison
**MainCrafts Technology — AI/ML Internship, Task 1 & Task 2**

## 📋 Objective
This project covers a full, progressive machine learning workflow on the California Housing dataset:

- **Task 1** builds a baseline regression model, covering the fundamentals: data loading, exploratory data analysis, preprocessing, training, evaluation, and an interactive prediction interface.
- **Task 2** extends this into a professional ML workflow: proper feature scaling, training multiple algorithms, comparing them using standardized metrics, checking for overfitting, and selecting the best model programmatically rather than by assumption.

Together, these tasks demonstrate the complete lifecycle of a regression project — from a first working model to a justified, optimized final model.

## 📊 Dataset
The California Housing dataset (built into scikit-learn) contains 20,640 rows and 9 columns describing housing and demographic characteristics across California districts, based on 1990 U.S. Census data.

**Target variable:** `MedHouseVal` — median house value for a district (in units of $100,000)

**Input features:**
| Feature | Description |
|---|---|
| MedInc | Median income in the district |
| HouseAge | Median house age in the district |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | District population |
| AveOccup | Average household occupancy |
| Latitude | District latitude |
| Longitude | District longitude |

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data handling:** pandas, numpy
- **Modeling:** scikit-learn — LinearRegression, Ridge, DecisionTreeRegressor, StandardScaler, train_test_split, evaluation metrics
- **Visualization:** matplotlib, seaborn
- **Interactivity:** ipywidgets
- **Model persistence:** joblib / pickle
- **Environment:** Jupyter Notebook

## ⚙️ Setup & How to Run
```bash
# Clone the repository
git clone https://github.com/DimpalTamta/california-housing-price-prediction.git
cd california-housing-price-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn ipywidgets joblib jupyter

# Launch Jupyter and run either notebook
jupyter notebook
```
No external dataset download is required — the California Housing dataset loads directly through scikit-learn.

---

## Task 1 — Baseline Linear Regression

### Objective
Establish a working end-to-end ML pipeline and a baseline model to compare all future improvements against.

### 🔍 Workflow
1. **Data Loading** — Loaded the dataset via `sklearn.datasets.fetch_california_housing` and combined features and target into a single DataFrame.
2. **Exploratory Data Analysis (EDA)** — Checked for missing values and data types, examined the distribution of each feature and the target variable, and built a correlation heatmap to identify which features relate most strongly to house price.
3. **Data Preparation** — Separated features (X) from the target (y) and split the data into training (80%) and test (20%) sets using `train_test_split`.
4. **Model Training** — Trained a `LinearRegression` model on the training set.
5. **Evaluation** — Measured performance on the held-out test set using MAE, RMSE, and R² score.
6. **Visualization** — Plotted Actual vs Predicted values to visually assess prediction accuracy, and examined residuals to check for systematic bias.
7. **Deployment** — Saved the trained model as a `.pkl` file using pickle, and built an interactive `ipywidgets`-based UI so a user can input neighborhood characteristics and get an instant price prediction.

### 📈 Results
| Metric | Value | Meaning |
|---|---|---|
| MAE | 0.533 | Average absolute prediction error (~$53,300) |
| RMSE | 0.746 | Root mean squared error (~$74,600); penalizes larger errors more heavily than MAE |
| R² Score | 0.576 | The model explains ~57.6% of the variance in house prices |

**Key finding:** Median Income (`MedInc`) is by far the strongest predictor of house value (correlation ≈ 0.688), which aligns with real-world intuition — wealthier districts tend to have higher property values.

**Limitation identified:** House values in the dataset are capped at $500,000, which distorts predictions and error metrics for higher-value districts — this was flagged as an area for future improvement.

### 🖥️ Interactive Prediction UI
An interactive widget-based interface lets a user enter neighborhood details — income, house age, average rooms/bedrooms, population, occupancy, and location — and instantly receive a predicted median house price.

*Example: median income $50,000, house age 25 years, 6 average rooms, 1 average bedroom, population 1,000, 3 people per household, near Los Angeles → predicted value ≈ $239,840.67*

---

## Task 2 — Feature Engineering, Model Optimization & Performance Comparison

### Objective
Move beyond a single baseline model to reflect how ML engineers actually work in practice: preprocessing data correctly, training and comparing multiple algorithms, and selecting a final model based on measurable evidence rather than a single attempt.

### 🔍 Workflow
1. **Train-Test Split First** — The dataset was split into training and test sets *before* any preprocessing, to prevent test-set statistics from leaking into the scaler (a common and easy-to-miss mistake).
2. **Feature Scaling** — Applied `StandardScaler`, fit only on the training data and then used to transform both the training and test sets, so every feature is on a comparable scale (mean 0, standard deviation 1).
3. **Multi-Model Training** — Trained three regression models on identical, scaled data:
   - **Linear Regression** — the baseline model, carried over from Task 1
   - **Ridge Regression** (alpha = 1.0) — linear regression with an L2 penalty, intended to reduce overfitting by shrinking coefficients
   - **Decision Tree Regressor** (max_depth = 5) — a non-linear model capable of capturing relationships a straight line cannot
4. **Evaluation** — Each model was scored on the test set using RMSE and R². Train-set R² was also recorded for every model, to check for overfitting by comparing it against test R².
5. **Model Selection** — Rather than assuming which model would perform best, the final model was selected programmatically — the model with the lowest test RMSE.
6. **Visual Performance Validation** — Plotted Actual vs Predicted values for the selected model to visually confirm its accuracy.

### 📈 Results
| Model | Train R² | Test R² | RMSE |
|---|---|---|---|
| **Decision Tree** | 0.6377 | 0.5997 | 0.7242 |
| Ridge Regression | 0.6126 | 0.5758 | 0.7456 |
| Linear Regression | 0.6126 | 0.5758 | 0.7456 |

**Selected model: Decision Tree Regressor (max_depth = 5)** — achieved the lowest RMSE and highest test R² of the three models.

**Overfitting check:** All three models show a small, similar gap between train and test R² (~0.04), meaning none of them overfit the training data. The Decision Tree's constrained depth kept it simple enough to generalize while still capturing non-linear structure.

**Key finding:** The Decision Tree's ~4% improvement in R² over Linear/Ridge Regression (0.600 vs 0.576) shows that house prices depend on non-linear relationships — such as income thresholds or geographic clustering — that a linear model structurally cannot represent. Feature scaling made little difference to the linear models' performance in this case, since Linear Regression's predictions are scale-invariant by construction and the dataset already has one dominant, strongly linear feature (median income) that a light Ridge penalty (alpha = 1.0) does little to regularize.

---

## 📁 Repository Contents
| File | Description |
|---|---|
| `task1/Task1_ml_linear_regression.ipynb` | Task 1 notebook — data loading, EDA, baseline Linear Regression, evaluation, interactive UI |
| `task1/House_Price_Estimator.docx` | Task 1 written report — EDA findings, model details, metrics, visualizations, conclusions |
| `task1/house_price_model.pkl` | Task 1 saved Linear Regression model |
| `task2/AI_ML_Task2_Model_Comparison.ipynb` | Task 2 notebook — scaling, multi-model training, comparison, best-model selection |
| `task2/AI_ML_Task2_Report.pdf` | Task 2 written report — methodology, results table, overfitting analysis, conclusions |
| `task2/best_model.pkl` | Task 2 saved best-performing model (Decision Tree) |
| `task2/scaler.pkl` | Task 2 saved StandardScaler, needed to preprocess new inputs consistently with training |

## 💡 Further Improvement Ideas
- **Ensemble models** — Random Forest or Gradient Boosting/XGBoost typically reach R² of 0.75–0.85+ on this dataset by combining many trees
- **Hyperparameter tuning** — use `GridSearchCV` or `RandomizedSearchCV` with cross-validation to tune Decision Tree depth, Ridge alpha, etc., instead of fixed values
- **Feature engineering** — derived features such as rooms-per-household, bedrooms-per-room, or distance to the nearest major city could give linear models more useful signal
- **Cross-validation** — k-fold cross-validation would give a more robust performance estimate than a single 80/20 split
- **Handling the price cap** — the dataset's $500,000 ceiling on house values distorts error metrics at the top of the price range and could be addressed with capped-value handling or a different target transformation

## 🙋 About
This project was completed as part of the AI/ML Internship program at MainCrafts Technology, covering Task 1 (baseline model) and Task 2 (model optimization and comparison).

**Website:** maincrafts.com
