import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric (if not done already)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})


# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Preprocessing
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


# 4. Model definitions + hyperparameters
models = {
    "logreg": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs"],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'model__n_estimators':[100,200],
            'model__max_depth':[5,10,None],
            'model__min_samples_split':[2,5]
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5],
        },
    },
}

best_models = {}


# 5. Train models with GridSearchCV
for name, m in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", m["model"])
    ])
    
    grid = GridSearchCV(pipe, m["params"], cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_pred = grid.predict(X_test)
    roc = roc_auc_score(y_test, grid.predict_proba(X_test)[:,1])
    print(f"\n{name.upper()} - ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred))
    
    best_models[name] = grid.best_estimator_


# 6. Save the best model-
# Chose the best based on ROC-AUC
winner_model = best_models["logreg"]
joblib.dump(winner_model, "churn_model.pkl")
print("\nFinal model saved to churn_model.pkl")
