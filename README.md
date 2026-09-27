# House Price Prediction — Linear Regression to Model Comparison
**MainCrafts Technology — AI/ML Internship, Task 1 & Task 2**

## 📋 Overview
This project implements an end-to-end machine learning workflow for predicting median house values in California, built in two progressive stages as part of an AI/ML internship program.

- **Task 1** establishes a baseline using a single Linear Regression model, covering the fundamentals: data loading, exploratory data analysis, training, evaluation, and an interactive prediction interface.
- **Task 2** builds on this baseline with a more rigorous, industry-aligned workflow: proper feature scaling, training and objectively comparing multiple regression algorithms, and selecting a final model based on measurable performance rather than assumption.

Together, the two tasks demonstrate the difference between a first-pass model and a properly validated, optimized one — a progression that mirrors how real-world ML projects evolve.

## 📊 Dataset
The California Housing dataset (built into scikit-learn) contains 20,640 rows and 9 columns describing housing characteristics across California districts, based on 1990 U.S. Census data.

**Target variable:** `MedHouseVal` — median house value for a district (in units of $100,000)

**Input features:**
| Feature | Description |
|---|---|
| MedInc | Median income in the district |
| HouseAge | Median house age in the district |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | District population |
| AveOccup | Average number of household members |
| Latitude | District latitude |
| Longitude | District longitude |

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data handling:** pandas, numpy
- **Modeling:** scikit-learn (LinearRegression, Ridge, DecisionTreeRegressor, StandardScaler, train_test_split, metrics)
- **Visualization:** matplotlib, seaborn
- **Interactivity:** ipywidgets
- **Persistence:** joblib / pickle

## ⚙️ Setup & Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd house-price-prediction-linear-regression

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn ipywidgets joblib jupyter
```

## ▶️ How to Run
```bash
# Task 1
jupyter notebook task1/Task1_ml_linear_regression.ipynb

# Task 2
jupyter notebook task2/AI_ML_Task2_Model_Comparison.ipynb
```
Run all cells top to bottom. The California Housing dataset is fetched automatically via scikit-learn — no manual download required.

---

## Task 1 — Baseline Linear Regression

### 🎯 Goal
Train a simple, interpretable model to establish a performance baseline before attempting any optimization.

### 🔍 Workflow
1. **Data Loading** — Loaded the dataset via `sklearn.datasets.fetch_california_housing`
2. **Exploratory Data Analysis** — Checked for missing values, examined feature distributions, and built a correlation heatmap to identify which features relate most strongly to house price
3. **Data Preparation** — Split into training (80%) and test (20%) sets
4. **Model Training** — Trained a `LinearRegression` model on the raw (unscaled) features
5. **Evaluation** — Assessed performance using MAE, RMSE, and R² score
6. **Visualization** — Plotted Actual vs Predicted values and residual diagnostics to visually inspect prediction quality
7. **Deployment** — Saved the trained model as a pickle file and built an interactive prediction UI using `ipywidgets`

### 📈 Results
| Metric | Value | Meaning |
|---|---|---|
| MAE | 0.533 | Average absolute prediction error (~$53,300) |
| RMSE | 0.746 | Root mean squared error (~$74,600) |
| R² Score | 0.576 | Model explains ~57.6% of the variance in house prices |

**Key finding:** Median Income (`MedInc`) is by far the strongest predictor of house value (correlation = 0.688), consistent with real-world intuition — wealthier districts tend to have higher-value homes.

**Limitation identified:** House values in the dataset are artificially capped at $500,000, which distorts prediction accuracy at the high end of the price range — a factor addressed in later improvement suggestions.

### 🖥️ Interactive Prediction UI
An interactive widget-based UI lets users enter neighborhood details (income, house age, rooms, location, etc.) and instantly get a predicted median house price, making the model tangible beyond just accuracy metrics.

*Example: median income $50,000, house age 25 years, 6 average rooms, 1 average bedroom, population 1,000, 3 people per household, near Los Angeles → predicted value ≈ $239,840.67*

---

## Task 2 — Feature Engineering, Model Optimization & Comparison

### 🎯 Goal
Move beyond a single model attempt by applying proper preprocessing and comparing multiple algorithms objectively — the way ML models are actually refined in professional settings.

### 🔍 Workflow
1. **Train-Test Split First** — The dataset was split into training and test sets *before* any preprocessing, so that no information from the test set leaks into the feature scaler (a common data leakage pitfall)
2. **Feature Scaling** — Applied `StandardScaler`, fitting it on the training data only, then transforming both sets. This ensures all features contribute fairly regardless of their original numeric range
3. **Multi-Model Training** — Trained and compared three regression models on identical, scaled data:
   - **Linear Regression** — baseline, for direct comparison against Task 1
   - **Ridge Regression** (alpha = 1.0) — linear model with an L2 penalty to reduce coefficient magnitude and control overfitting
   - **Decision Tree Regressor** (max_depth = 5) — a non-linear model capable of capturing relationships a straight line cannot
4. **Evaluation** — Compared RMSE and Test R² across all three models, and recorded Train R² to check for overfitting
5. **Model Selection** — Selected the best-performing model programmatically (lowest test RMSE), rather than assuming a winner in advance
6. **Visualization** — Plotted Actual vs Predicted values for the selected model to visually validate performance

### 📈 Results
| Model | Train R² | Test R² | RMSE |
|---|---|---|---|
| **Decision Tree** | 0.6377 | 0.5997 | 0.7242 |
| Ridge Regression | 0.6126 | 0.5758 | 0.7456 |
| Linear Regression | 0.6126 | 0.5758 | 0.7456 |

**Selected model: Decision Tree Regressor (max_depth = 5)**

### 🧠 Overfitting Check
| Model | Train R² | Test R² | Gap |
|---|---|---|---|
| Decision Tree | 0.638 | 0.600 | 0.038 |
| Ridge Regression | 0.613 | 0.576 | 0.037 |
| Linear Regression | 0.613 | 0.576 | 0.037 |

All three models show a small, comparable train-test gap, meaning none of them are overfitting. Constraining the Decision Tree's depth to 5 keeps it simple enough to generalize, while still letting it capture non-linear patterns — such as income thresholds and geographic clustering — that Linear and Ridge Regression cannot represent.

### 💬 Key Findings
- The Decision Tree outperformed both linear models by roughly 4% in test R² (0.600 vs 0.576), confirming that house prices depend on non-linear relationships among features
- Feature scaling had negligible effect on Linear and Ridge Regression in this case: Linear Regression is mathematically scale-invariant, and a Ridge penalty of 1.0 is too light to meaningfully affect a dataset already dominated by one strong linear predictor (`MedInc`)
- Scaling remains good practice regardless, since it's required for many other algorithms (e.g. KNN, SVM, gradient-descent–based models) not used here

---

## 📁 Repository Contents
| File | Description |
|---|---|
| `task1/Task1_ml_linear_regression.ipynb` | Task 1 notebook — baseline Linear Regression, EDA, evaluation, UI |
| `task1/House_Price_Estimator.docx` | Task 1 written report (EDA, model, metrics, visualizations, conclusions) |
| `task1/house_price_model.pkl` | Task 1 saved Linear Regression model |
| `task2/AI_ML_Task2_Model_Comparison.ipynb` | Task 2 notebook — scaling, multi-model training and comparison |
| `task2/AI_ML_Task2_Report.pdf` | Task 2 written report (methodology, results, model selection, conclusions) |
| `task2/best_model.pkl` | Task 2 saved best-performing model (Decision Tree) |
| `task2/scaler.pkl` | Task 2 saved `StandardScaler`, needed to preprocess new inputs consistently |

## 💡 Further Improvement Ideas
- **Ensemble models** — Random Forest or Gradient Boosting (XGBoost/LightGBM) typically reach R² of 0.75–0.85+ on this dataset by combining many trees
- **Hyperparameter tuning** — use `GridSearchCV` or `RandomizedSearchCV` with cross-validation to tune Decision Tree depth, Ridge alpha, etc., instead of fixed values
- **Feature engineering** — derived features such as rooms-per-household, bedrooms-per-room, or distance to major cities could give linear models more useful signal
- **Cross-validation** — k-fold cross-validation would give a more robust performance estimate than a single 80/20 split
- **Price-cap handling** — the dataset's artificial $500,000 cap on house values distorts error metrics at the top of the price range; this could be addressed by removing or flagging capped rows

## 🙋 About
This project was completed as part of the AI/ML Internship program at MainCrafts Technology, progressing from foundational model training (Task 1) to a comparative, optimization-focused workflow (Task 2).

**Website:** maincrafts.com
