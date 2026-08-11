from flask import Flask, render_template, send_from_directory
from src.data.load_data import load_data, get_summary
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset")
def dataset():
    # Load your dataset
    df = load_data()
    summary = get_summary(df)

    # Pass summary + first rows to template
    return render_template(
        "load_dataset.html",
        summary=summary,
        first_rows=df.head().to_html(index=False)
    )

@app.route("/eda")
def eda():
    return render_template("eda.html")

# Serve images from results/images
@app.route("/results/charts/<path:filename>")
def results_charts(filename):
    # Absolute path to Charts directory
    base_dir = r"C:\Users\asolo\OneDrive\Documents\ML\Placement-Prediction-System\app\static\Charts"
    return send_from_directory(base_dir, filename)


if __name__ == "__main__":
    app.run(debug=True)
