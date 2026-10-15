import argparse
import os
import warnings
import io

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, f1_score
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

FEATURES = [
    "Growth", "Soil", "Sunlight", "Watering",
    "Fertilization Type"
]
OUT_DIR = "outputs"


# 1. LOAD AND CLEAN DATA
def load_and_clean(path):
    lines = []

    with open(path, "rb") as file:
        for raw in file.read().split(b"\n"):
            try:
                lines.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append(raw.decode("cp1252"))

    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = df.columns.str.strip()

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    df["Plant Name"] = (
        df["Plant Name"]
        .str.replace("\u2019", "'", regex=False)
        .str.replace("\u2018", "'", regex=False)
    )

    for c in ["Growth", "Soil", "Sunlight", "Fertilization Type"]:
        df[c] = df[c].str.lower()

    def water_group(t):
        t = t.lower()
        if "dry" in t:
            return "let dry between"
        if "moist" in t:
            return "keep moist"
        if "weekly" in t:
            return "weekly"
        return "regular"

    df["Watering"] = df["Watering"].apply(water_group)
    df = df.drop_duplicates().reset_index(drop=True)

    return df


# 2. EDA GRAPHS
def make_eda_graphs(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, col in zip(axes.ravel(), FEATURES):
        order = df[col].value_counts().index
        sns.countplot(
            data=df, y=col, order=order,
            ax=ax, palette="viridis"
        )
        ax.set_title(f"Distribution of {col}")

    top = df["Plant Name"].value_counts().head(8)
    sns.barplot(
        x=top.values, y=top.index,
        ax=axes[1, 2], palette="mako"
    )
    axes[1, 2].set_title("Plant names with most variants")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_distributions.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.heatmap(
        pd.crosstab(df["Soil"], df["Sunlight"]),
        annot=True, fmt="d", cmap="YlGn"
    )
    plt.title("Soil vs Sunlight")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_soil_vs_sunlight.png", dpi=130)
    plt.close()

    def cramers_v(a, b):
        ct = pd.crosstab(a, b)
        expected = (
            ct.sum(1).values[:, None]
            * ct.sum(0).values / ct.values.sum()
        )
        chi2 = (((ct - expected) ** 2) / expected).values.sum()
        n = ct.values.sum()
        return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

    m = pd.DataFrame(index=FEATURES, columns=FEATURES, dtype=float)

    for a in FEATURES:
        for b in FEATURES:
            m.loc[a, b] = (
                1.0 if a == b else cramers_v(df[a], df[b])
            )

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        m, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=0, vmax=1
    )
    plt.title("Feature Association")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_feature_association.png", dpi=130)
    plt.close()


# 3. TRAIN AND COMPARE MODELS
def train_models(df):
    X_cols = ["Growth", "Soil", "Sunlight", "Watering"]
    X = df[X_cols]
    y = df["Fertilization Type"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), X_cols)
    ])

    models = {
        "Baseline (majority)": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42
        )
    }

    results = []
    fitted = {}

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        cv = cross_val_score(
            pipe, X_tr, y_tr, cv=5, scoring="accuracy"
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        results.append({
            "Model": name,
            "CV accuracy": cv.mean(),
            "Test accuracy": accuracy_score(y_te, pred),
            "Test F1 (weighted)": f1_score(
                y_te, pred, average="weighted"
            )
        })

        fitted[name] = pipe

    res = pd.DataFrame(results).sort_values(
        "Test accuracy", ascending=False
    )

    print("\n=== MODEL COMPARISON ===")
    print(res.round(3).to_string(index=False))

    res_m = res.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=res_m, x="Model", y="Score",
        hue="Metric", palette="Set2"
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_model_comparison.png", dpi=130)
    plt.close()

    best_name = res.iloc[0]["Model"]
    best = fitted[best_name]
    pred = best.predict(X_te)

    print("\nBest model:", best_name)
    print(classification_report(y_te, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, ax=ax, cmap="Blues",
        xticks_rotation=30
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_confusion_matrix.png", dpi=130)
    plt.close()

    rf = fitted["Random Forest"]
    names = rf.named_steps["pre"].get_feature_names_out()

    imp = pd.Series(
        rf.named_steps["clf"].feature_importances_,
        index=names
    ).sort_values().tail(12)

    plt.figure(figsize=(8, 5))
    imp.plot.barh(color="seagreen")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_feature_importance.png", dpi=130)
    plt.close()

    joblib.dump(best, f"{OUT_DIR}/best_model.joblib")

    return best, best_name, res


# 4. PLANT RECOMMENDER
class PlantRecommender:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.enc = OneHotEncoder(handle_unknown="ignore")
        self.matrix = self.enc.fit_transform(self.df[FEATURES])

    def recommend(self, top_n=5, **prefs):
        query = {f: "" for f in FEATURES}

        for k, v in prefs.items():
            col = k.replace("_", " ").title()
            if col.startswith("Fert"):
                col = "Fertilization Type"

            if col not in FEATURES:
                raise ValueError(f"Unknown feature: {k}")

            query[col] = str(v).lower()

        q = self.enc.transform(pd.DataFrame([query]))

        mask = np.zeros(self.matrix.shape[1])
        offset = 0

        for f, cats in zip(FEATURES, self.enc.categories_):
            if query[f]:
                mask[offset:offset + len(cats)] = 1
            offset += len(cats)

        a = self.matrix.multiply(mask).tocsr()
        b = q.multiply(mask).tocsr()

        scores = cosine_similarity(a, b).ravel()

        out = self.df.assign(Match=(scores * 100).round(1))
        out = out.sort_values(
            ["Match", "Plant Name"],
            ascending=[False, True]
        )

        return (
            out.drop_duplicates("Plant Name")
            .head(top_n).reset_index(drop=True)
        )

    def similar_to(self, name, top_n=5):
        hits = self.df.index[
            self.df["Plant Name"].str.lower() == name.lower()
        ]

        if len(hits) == 0:
            guess = self.df[
                self.df["Plant Name"].str.lower().str.contains(
                    name.lower(), regex=False
                )
            ]
            raise ValueError(
                f"'{name}' not found. Did you mean: "
                f"{sorted(guess['Plant Name'].unique())[:5]}"
            )

        scores = cosine_similarity(
            self.matrix[hits[0]], self.matrix
        ).ravel()

        out = self.df.assign(Match=(scores * 100).round(1))
        out = out[
            out["Plant Name"].str.lower() != name.lower()
        ]
        out = out.sort_values(
            ["Match", "Plant Name"],
            ascending=[False, True]
        )

        return (
            out.drop_duplicates("Plant Name")
            .head(top_n).reset_index(drop=True)
        )


# 5. RECOMMENDATION GRAPHS
def plot_recommendations(rec_df, title, fname):
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=rec_df, x="Match", y="Plant Name",
        palette="crest"
    )
    plt.xlim(0, 105)
    plt.title(title)
    plt.xlabel("Match score (%)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=130)
    plt.close()


def plot_similarity_heatmap(df, rec):
    names = [
        "Aloe Vera", "Basil", "Snake Plant", "Lavender",
        "Cactus", "Rosemary", "Marigold", "Sunflower",
        "Petunia", "Sage"
    ]

    idx = [
        df.index[df["Plant Name"] == n][0]
        for n in names
        if (df["Plant Name"] == n).any()
    ]

    if not idx:
        return

    sim = cosine_similarity(rec.matrix[idx])
    labels = df.loc[idx, "Plant Name"]

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        sim, xticklabels=labels, yticklabels=labels,
        annot=True, fmt=".2f", cmap="YlOrRd"
    )
    plt.title("Plant-to-Plant Similarity")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_similarity_heatmap.png", dpi=130)
    plt.close()


def print_table(df):
    print(df[[
        "Plant Name", "Growth", "Soil", "Sunlight",
        "Watering", "Fertilization Type", "Match"
    ]].to_string(index=False))


# 6. INTERACTIVE MODE
def interactive(df, rec, model):
    print("\n=== INTERACTIVE MODE ===")
    options = {f: sorted(df[f].unique()) for f in FEATURES}

    while True:
        print(
            "\n1) Recommend by conditions"
            "\n2) Plants similar to a plant"
            "\n3) Predict fertilizer"
            "\nq) Quit"
        )

        ch = input("Choose: ").strip().lower()

        if ch == "q":
            break

        elif ch == "1":
            prefs = {}

            for f in FEATURES:
                v = input(f"{f} {options[f]}: ").strip().lower()
                if v:
                    prefs[f] = v

            if not prefs:
                print("Give at least one preference.")
                continue

            res = rec.recommend(top_n=5, **prefs)
            print_table(res)

            plot_recommendations(
                res, "Your Recommendations",
                "09_my_recommendations.png"
            )

        elif ch == "2":
            name = input("Plant name: ").strip()

            try:
                res = rec.similar_to(name, top_n=5)
                print_table(res)
            except ValueError as e:
                print(e)

        elif ch == "3":
            row = {}

            for f in ["Growth", "Soil", "Sunlight", "Watering"]:
                row[f] = input(
                    f"{f} {options[f]}: "
                ).strip().lower()

            print(
                "Predicted fertilizer:",
                model.predict(pd.DataFrame([row]))[0]
            )


# MAIN
def main():
    ap = argparse.ArgumentParser()

    # DATASET FILE PATH
    ap.add_argument(
        "--csv",
        default=r"C:\Users\Mansi\Downloads\archive\plants.csv"
    )

    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # READ THE DATASET
    df = load_and_clean(args.csv)

    print(
        f"Clean data: {df.shape[0]} rows, "
        f"{df['Plant Name'].nunique()} unique plants"
    )
    print(df.head())

    make_eda_graphs(df)

    model, best_name, results = train_models(df)

    rec = PlantRecommender(df)
    plot_similarity_heatmap(df, rec)

    print("\n=== DEMO 1: Sunny + Sandy Soil ===")

    r1 = rec.recommend(
        top_n=6,
        sunlight="full sunlight",
        soil="sandy",
        watering="let dry between"
    )

    print_table(r1)

    plot_recommendations(
        r1,
        "Top Picks: Full Sun, Sandy Soil",
        "07_demo_recommendations.png"
    )

    print("\n=== DEMO 2: Plants Similar to Basil ===")
    print_table(rec.similar_to("Basil", top_n=6))

    print(f"\nGraphs and saved model are in '{OUT_DIR}'.")

    if args.interactive:
        interactive(df, rec, model)

if __name__ == "__main__":
    main()