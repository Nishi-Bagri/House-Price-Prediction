# House-Price-Prediction
End-to-end House Price Prediction project using Machine Learning. Includes data cleaning, EDA, preprocessing with pipelines, and regression modeling (Linear, Ridge, Lasso). Offers insights on key factors affecting prices and achieves an R² score of 0.54 using Linear Regression.
🏡 House Price Prediction Using Regression
End-to-End Machine Learning Project (Python, Scikit-learn)
📌 Project Overview

This project builds an end-to-end Machine Learning regression model to predict house prices using multiple property features such as area, bedrooms, bathrooms, location, mainroad access, basement, guestroom, furnishing status, and more.

The workflow includes:

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Preprocessing using Pipelines

Model Training (Linear, Ridge, Lasso Regression)

Model Evaluation

Insights & Conclusion

📂 Dataset

Rows: 500

Columns: 14

Target Variable: price

Features:

Numerical: area, bedroom, stories, parking, bathrooms

Categorical: mainroad, guestroom, basement, hotwaterheating,
air-conditioning, prefarea, furnishing status, town

Contains missing values in:

guestroom, basement, furnishing status

🔍 Exploratory Data Analysis (EDA)
Key Observations:

Area has the strongest positive correlation with price.

Houses with more bedrooms and bathrooms cost more.

Mainroad access significantly increases price.

Furnished houses generally have higher prices.

Town (location) plays a major role in price variation.

Plots Included:

Histograms for numerical features

Countplots for categorical features

Scatter plot (price vs area)

Boxplots by categories

Correlation heatmap

🛠️ Data Preprocessing

✔ Missing values imputed using most frequent category
✔ Categorical variables encoded using OneHotEncoder
✔ Numerical columns scaled using StandardScaler
✔ Preprocessing handled via ColumnTransformer + Pipeline

Sample Preprocessing Code:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ]
)

🤖 Modeling

Three regression models were trained:

✔ 1. Linear Regression (Best model)
✔ 2. Ridge Regression
✔ 3. Lasso Regression (did not converge; not suitable)**

All models were embedded inside a pipeline:

model_lr = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LinearRegression())
])

📈 Model Performance
Linear Regression Results
Metric	Value
MAE	23.55 lakhs
RMSE	36.07 lakhs
R² Score	0.5418
Interpretation:

The model explains 54% of the variation in house prices.

Reasonable performance for a mixed numeric + categorical dataset.

Error margin is acceptable considering price ranges (₹5 lakh – ₹5 crore).

Ridge Regression:

Same performance as Linear Regression.

Lasso Regression:

Failed to converge → dataset unsuitable for L1 regularization.

🧠 Key Insights

Area is the strongest predictor of house price.

Location (town) significantly affects prices.

Houses on main roads, with guest rooms, or furnishing tend to be expensive.

Regularization (Ridge, Lasso) does not improve performance.

Linear Regression is the best model for this dataset.

🏁 Conclusion

The Linear Regression model provides a stable and interpretable solution for predicting house prices with an R² score of 0.54.
This project demonstrates complete ML workflow skills including data cleaning, EDA, preprocessing, regression modeling, evaluation, and insights — making it ideal for interviews and portfolio use.

📜 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

⭐ Author

Nishi Bagri
Aspiring Data Scientist / Machine Learning Engineer

🌟 House Price Prediction Using Machine Learning
📊 End-to-End Regression Project (Python, Scikit-learn)

📚 Table of Contents

Project Overview

Tech Stack

Dataset Description

Exploratory Data Analysis

Data Preprocessing

Modeling

Results

Key Insights

Project Structure

How to Run

Future Improvements

Author

License

🧾 Project Overview

This project predicts house prices based on various property features using supervised machine learning techniques.
The goal is to build a reliable regression model and understand which features influence pricing the most.

The project includes:

Complete EDA

Feature engineering

Preprocessing using ColumnTransformer

Multiple ML models (Linear, Ridge, Lasso)

Evaluation and comparison

Final insights for decision-making

🧰 Tech Stack
Category	Tools Used
Programming	Python
Libraries	Pandas, NumPy, Matplotlib, Seaborn
ML Framework	Scikit-learn
Model types	Linear, Ridge, Lasso Regression
Environment	Jupyter Notebook

📂 Dataset Description

The dataset contains 500 rows and 14 features.

🔹 Numerical Features

area

bedroom

stories

parking

bathrooms

price (target)

🔹 Categorical Features

mainroad

guestroom

basement

hotwaterheating

air-conditioning

prefarea

furnishing status

town

🔸 Missing Values

guestroom → 15 missing

basement → 15 missing

furnishing status → 15 missing

🔍 Exploratory Data Analysis

EDA included:

📊 Visualizations:

Histograms

Countplots

Scatter plot (area vs price)

Boxplots

Correlation heatmap

📝 EDA Summary:

Price increases with area, bedrooms, stories

Main road access increases house value

Furnished homes usually cost more

Strong relationship: area → price

Town plays a major role in price distribution

🛠️ Data Preprocessing

Preprocessing was automated with ColumnTransformer + Pipeline.

🔧 Steps:

✔ Missing values imputed (most frequent)
✔ One-Hot Encoding for categorical features
✔ Standard scaling for numerical features
✔ Train-test split (80-20)

🧱 Pipeline Example:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

🤖 Modeling

Three regression models were trained:

Linear Regression

Ridge Regression

Lasso Regression (did not converge)

📈 Results
Linear Regression (Best Model)
Metric	Value
MAE	23.55 lakhs
RMSE	36.07 lakhs
R² Score	0.5418
Ridge Regression

Similar performance to Linear Regression

No improvement in accuracy

Lasso Regression

❌ Did not converge even after tuning → not suitable

🧠 Key Insights

Area is the strongest predictor of house price

Town (location) highly affects price

Homes with guest rooms, mainroad access, furnishing are more expensive

No strong multicollinearity → linear regression performs well

Regularization does not improve performance

📁 Project Structure
House-Price-Prediction/
│
├── data/
│   └── Newhousing1.csv
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── models/
│   └── house_price_model.pkl
│
├── README.md
└── requirements.txt
