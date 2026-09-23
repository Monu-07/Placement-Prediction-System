from flask import Flask, render_template, send_from_directory
from src.data.load_data import load_data, get_summary
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data,
)
import os
import joblib
import json

app = Flask(__name__)

# Root of the project (one level above /App)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_INFO = [
    {
        "key": "logistic_regression",
        "name": "Logistic Regression",
        "type": "Classification",
        "file": "logistic_regression.pkl",
        "chart": None,
        "description": "A linear baseline model estimating placement probability using weighted numerical and encoded categorical features.",
        "key_params": ["C", "penalty", "solver", "max_iter"],
    },
    {
        "key": "decision_tree",
        "name": "Decision Tree",
        "type": "Classification",
        "file": None,
        "chart": "decision_tree.png",
        "description": "An ID3-style tree (entropy criterion, max depth 5) that splits students into groups based on features reducing uncertainty.",
        "key_params": ["criterion", "max_depth"],
    },
    {
        "key": "random_tree",
        "name": "Random Tree",
        "type": "Classification",
        "file": "random_tree.pkl",
        "chart": None,
        "description": "A single decision tree trained with random splitter, used as a variance baseline for Random Forest.",
        "key_params": ["criterion", "splitter", "max_depth"],
    },
    {
        "key": "random_forest",
        "name": "Random Forest",
        "type": "Classification",
        "file": "random_forest.pkl",
        "chart": "random_forest_importances.png",
        "description": "An ensemble of 100 entropy-based decision trees that averages predictions to improve accuracy and reduce overfitting.",
        "key_params": ["n_estimators", "criterion", "max_depth", "oob_score"],
    },
    {
        "key": "gradient_boosting",
        "name": "Gradient Boosting",
        "type": "Classification",
        "file": None,
        "chart": None,
        "description": "An ensemble that builds trees sequentially, correcting previous errors.",
        "key_params": ["n_estimators", "learning_rate", "max_depth"],
    },
    {
        "key": "adaboost",
        "name": "AdaBoost",
        "type": "Classification",
        "file": None,
        "chart": None,
        "description": "Adaptive boosting ensemble that re-weights misclassified students so later learners focus on harder cases.",
        "key_params": ["n_estimators", "learning_rate"],
    },
    {
        "key": "kmeans",
        "name": "K-Means Clustering",
        "type": "Unsupervised / Clustering",
        "file": None,
        "chart": "kmeans_clusters.png",
        "description": "Unsupervised algorithm (k=3) grouping students with similar profiles.",
        "key_params": ["n_clusters", "n_init"],
    },
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset")
def dataset():
    df = load_data()
    summary = get_summary(df)
    return render_template(
        "load_dataset.html",
        summary=summary,
        first_rows=df.head().to_html(index=False)
    )

@app.route("/eda")
def eda():
    return render_template("eda.html")

@app.route("/preprocessing")
def preprocessing():
    try:
        df = load_data()
        raw_shape = df.shape

        X_train, X_test, y_train, y_test = split_data(
            df,
            drop_columns=["StudentID", "Salary Package", "IsAnomaly"],
        )
        split_train_shape = X_train.shape
        split_test_shape = X_test.shape

        numerical_features, categorical_features = identify_features(X_train)

        one_hot_features = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
        ordinal_features = ["CollegeTier", "CGPA_Tier"]

        X_train, X_test, _ = handle_missing_values(X_train, X_test, numerical_features)
        X_train, X_test, _ = standardize_data(X_train, X_test, numerical_features)
        X_train, X_test, _ = one_hot_encode_data(X_train, X_test, one_hot_features)
        X_train, X_test, _ = ordinal_encode_data(X_train, X_test, ordinal_features)

        preview = X_train.head().to_html(index=False, classes="preview-table")

        data = {
            "error": None,
            "raw_shape": raw_shape,
            "split_train_shape": split_train_shape,
            "split_test_shape": split_test_shape,
            "final_train_shape": X_train.shape,
            "final_test_shape": X_test.shape,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "one_hot_features": one_hot_features,
            "ordinal_features": ordinal_features,
            "preview": preview,
        }
    except Exception as e:
        data = {"error": str(e)}

    return render_template("preprocessing.html", data=data)

@app.route("/models")
def models():
    model_cards = []
    results = {}
    best_model_name = None

    # Load results.json if available
    results_path = os.path.join(BASE_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

        # Pick best model by highest accuracy
        accuracies = {k: v["accuracy"] for k, v in results.items() if v["accuracy"] is not None}
        if accuracies:
            best_model_name = max(accuracies, key=accuracies.get)

    for info in MODEL_INFO:
        card = dict(info)
        card["loaded"] = False
        card["params"] = {}

        if info["file"]:
            model_path = os.path.join(MODELS_DIR, info["file"])
            if os.path.exists(model_path):
                try:
                    estimator = joblib.load(model_path)
                    full_params = estimator.get_params()
                    card["params"] = {
                        k: full_params.get(k)
                        for k in info["key_params"]
                        if k in full_params
                    }
                    card["loaded"] = True
                except Exception as e:
                    card["load_error"] = str(e)

        model_cards.append(card)

    return render_template(
        "models.html",
        models=model_cards,
        results=results,
        best_model_name=best_model_name
    )

@app.route("/results/charts/<path:filename>")
def results_charts(filename):
    base_dir = r"/App/static/Charts"
    return send_from_directory(base_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)
