# House Price Prediction — Linear Regression to Model Comparison
**MainCrafts Technology — AI/ML Internship, Task 1 & Task 2**

## 📋 Objective
Build and evaluate machine learning models on the California Housing dataset, progressing from a single baseline model (Task 1) to a multi-model comparison with feature scaling and optimization (Task 2).

## 📊 Dataset
The California Housing dataset (built into scikit-learn) — 20,640 rows, 9 columns, describing housing characteristics across California districts. Goal: predict the median house value (`MedHouseVal`) for each district.

**Features used:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## 🛠️ Tech Stack
- Python
- pandas, numpy
- scikit-learn (LinearRegression, Ridge, DecisionTreeRegressor, StandardScaler, train_test_split, metrics)
- matplotlib, seaborn (visualization)
- ipywidgets (interactive UI)
- joblib (model persistence)

---

## Task 1 — Baseline Linear Regression

### 🔍 Workflow
1. **Data Loading** — Loaded the dataset via `sklearn.datasets.fetch_california_housing`
2. **EDA** — Checked for missing values, examined feature distributions, built a correlation heatmap
3. **Data Preparation** — Split into training (80%) and test (20%) sets
4. **Model Training** — Trained a `LinearRegression` model
5. **Evaluation** — Assessed performance using MAE, RMSE, and R² score
6. **Visualization** — Plotted Actual vs Predicted values and residual diagnostics
7. **Deployment** — Saved the trained model as a pickle file and built an interactive prediction UI using `ipywidgets`

### 📈 Results
| Metric | Value | Meaning |
|---|---|---|
| MAE | 0.533 | Average absolute prediction error (~$53,300) |
| RMSE | 0.746 | Root mean squared error (~$74,600) |
| R² Score | 0.576 | Model explains ~57.6% of the variance in house prices |

**Key finding:** Median Income (`MedInc`) is by far the strongest predictor of house value (correlation = 0.688), consistent with real-world intuition.

### 🖥️ Interactive Prediction UI
An interactive widget-based UI lets users enter neighborhood details (income, house age, rooms, location, etc.) and instantly get a predicted median house price.

*Example: median income $50,000, house age 25 years, 6 average rooms, 1 average bedroom, population 1,000, 3 people per household, near Los Angeles → predicted value ≈ $239,840.67*

---

## Task 2 — Feature Engineering, Model Optimization & Comparison

### 🔍 Workflow
1. **Train-Test Split First** — Split before any preprocessing, to avoid test-data leakage
2. **Feature Scaling** — Applied `StandardScaler`, fit on training data only
3. **Multi-Model Training** — Trained and compared 3 models:
   - Linear Regression (baseline)
   - Ridge Regression (alpha = 1.0, L2-regularized)
   - Decision Tree Regressor (max_depth = 5)
4. **Evaluation** — Compared RMSE, Test R², and Train R² (to check overfitting)
5. **Model Selection** — Selected the best model programmatically, based on lowest test RMSE
6. **Visualization** — Actual vs Predicted plot for the selected model

### 📈 Results
| Model | Train R² | Test R² | RMSE |
|---|---|---|---|
| **Decision Tree** | 0.6377 | 0.5997 | 0.7242 |
| Ridge Regression | 0.6126 | 0.5758 | 0.7456 |
| Linear Regression | 0.6126 | 0.5758 | 0.7456 |

**Selected model: Decision Tree Regressor (max_depth = 5)** — best RMSE and R² among the three, with no meaningful overfitting (train-test gap of ~0.04 across all models).

**Key finding:** The Decision Tree's ~4% R² improvement over Linear/Ridge Regression confirms house prices have non-linear structure that linear models under-fit. Feature scaling had negligible effect on Linear/Ridge here, since Linear Regression is scale-invariant and one feature (income) already dominates the signal.

---

## 📁 Repository Contents
| File | Description |
|---|---|
| `task1/Task1_ml_linear_regression.ipynb` | Task 1 notebook — baseline Linear Regression |
| `task1/House_Price_Estimator.docx` | Task 1 report (EDA, model, metrics, visualizations, conclusions) |
| `task1/house_price_model.pkl` | Task 1 saved Linear Regression model |
| `task2/AI_ML_Task2_Model_Comparison.ipynb` | Task 2 notebook — scaling, multi-model comparison |
| `task2/AI_ML_Task2_Report.pdf` | Task 2 report (methodology, results, model selection, conclusions) |
| `task2/best_model.pkl` | Task 2 saved best model (Decision Tree) |
| `task2/scaler.pkl` | Task 2 saved `StandardScaler` |

## 💡 Further Improvement Ideas
- Ensemble models (Random Forest, Gradient Boosting/XGBoost) — typically reach R² of 0.75–0.85+ on this dataset
- Hyperparameter tuning via `GridSearchCV` with cross-validation
- Feature engineering (e.g. rooms-per-household, distance to city center)
- k-fold cross-validation instead of a single train/test split
- Handling the artificially capped $500,000 house values, which distort error metrics at the high end

## 🙋 About
This project was completed as part of the AI/ML Internship program at MainCrafts Technology.

**Website:** maincrafts.com
