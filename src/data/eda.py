from statistics import correlation
from src.data.load_data import load_data
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

BASE_DIR = r"C:\Users\asolo\OneDrive\Documents\ML\Placement-Prediction-System\app\static\Charts"
os.makedirs(BASE_DIR, exist_ok=True)

os.makedirs(BASE_DIR, exist_ok=True)

def basic_eda(df):
    print("First Five Rows")
    print(df.head())
    print("Last Five Rows")
    print(df.tail())
    print("\n25 to 30 records:")
    print(df.iloc[24:30])
    print("Datatypes")
    print(df.dtypes)
    print("Complete Information")
    print(df.info())
    print("Duplicates")
    print(df.duplicated())
    print("Null Values")
    print(df.isnull().sum())
    print(df["PlacementStatus"].value_counts())
    count = df["PlacementStatus"].value_counts()
    plt.figure(figsize=(5,6))
    plt.bar(count.index,count.values)
    plt.title("Count of Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(os.path.join(BASE_DIR, "Placement_Status.png"))
    plt.show()

def univariate_eda(df):
    plt.figure(figsize=(5,6))
    plt.hist(df["CGPA"], bins=10)
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(BASE_DIR, "CGPA_Histogram.png"))
    plt.show()
    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6, 5))
    plt.pie(gendercount, labels=gendercount.index, autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of Gender")
    plt.xlabel("Gender")
    plt.savefig(os.path.join(BASE_DIR, "Gender_Count.png"))
    plt.show()

def bivariate(df):
    plt.figure(figsize = (6,5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"])
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(os.path.join(BASE_DIR, "CGPA_vs_Aptitude_Test_Score.png"))
    plt.show()
    plt.close()

    placed = df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed = df[df["PlacementStatus"] == 0]["CGPA"]
    plt.boxplot([placed, not_placed], label=["placed", "not_placed"])
    plt.title("CGPA vs PlacementStatus")
    plt.xlabel("PlacementStatus")
    plt.ylabel("CGPA")
    plt.savefig(os.path.join(BASE_DIR, "CGPA_vs_PlacementStatus.png"))
    plt.show()


def multivariate(df):
    data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]
    correlation=data.corr()
    plt.figure(figsize = (8,6))
    sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f")
    plt.title("correlation heatmap")
    plt.savefig(os.path.join(BASE_DIR, "correlation_heatmap.png"))
    plt.show()
    plt.close()

if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate_eda(df)
    bivariate(df)
    multivariate(df)
