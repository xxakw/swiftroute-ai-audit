from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "work" / "Train.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


def pct(series):
    return (series * 100).round(2)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    target = "Reached.on.Time_Y.N"
    target_label = {0: "On time (0)", 1: "Not on time (1)"}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    feature_numeric = [c for c in numeric_cols if c not in ["ID", target]]

    sns.set_theme(style="whitegrid", palette="Set2")

    # Target class balance.
    target_counts = df[target].value_counts().sort_index()
    target_pct = pct(target_counts / len(df))
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=[target_label[i] for i in target_counts.index],
        y=target_counts.values,
        hue=[target_label[i] for i in target_counts.index],
        legend=False,
    )
    ax.set_title("Target Class Balance")
    ax.set_xlabel("Delivery outcome")
    ax.set_ylabel("Shipments")
    for i, value in enumerate(target_counts.values):
        ax.text(i, value + 70, f"{value:,}\n({target_pct.iloc[i]}%)", ha="center")
    savefig(FIG_DIR / "target_class_balance.png")

    # Numeric univariate plots: histograms and boxplots.
    n = len(feature_numeric)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    for i, col in enumerate(feature_numeric):
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color="#4C78A8")
        axes[i, 0].set_title(f"{col} distribution")
        sns.boxplot(x=df[col], ax=axes[i, 1], color="#F58518")
        axes[i, 1].set_title(f"{col} boxplot")
    savefig(FIG_DIR / "numeric_univariate.png")

    # Categorical univariate plots.
    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(8, 4 * len(categorical_cols)))
    if len(categorical_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, categorical_cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, ax=ax, color="#54A24B")
        ax.set_title(f"{col} frequency")
        ax.set_xlabel(col)
        ax.set_ylabel("Shipments")
        for label in ax.get_xticklabels():
            label.set_rotation(0)
    savefig(FIG_DIR / "categorical_univariate.png")

    # Bivariate plots against target.
    bivariate_features = [
        "Discount_offered",
        "Weight_in_gms",
        "Product_importance",
        "Mode_of_Shipment",
        "Customer_care_calls",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()
    sns.boxplot(data=df, x=target, y="Discount_offered", ax=axes[0], palette="Set2")
    axes[0].set_title("Discount offered by delivery outcome")
    axes[0].set_xticklabels([target_label[0], target_label[1]])

    sns.boxplot(data=df, x=target, y="Weight_in_gms", ax=axes[1], palette="Set2")
    axes[1].set_title("Weight by delivery outcome")
    axes[1].set_xticklabels([target_label[0], target_label[1]])

    for ax, col in zip(axes[2:], bivariate_features[2:]):
        rate = df.groupby(col)[target].mean().sort_values(ascending=False)
        sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#E45756")
        ax.set_title(f"Not-on-time rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Not-on-time rate")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    savefig(FIG_DIR / "bivariate_target_relationships.png")

    # Correlation heatmap.
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap for Numeric Features")
    savefig(FIG_DIR / "numeric_correlation_heatmap.png")

    # Summary tables/statistics.
    dtypes = df.dtypes.astype(str).rename("dtype").to_frame()
    missing = df.isna().sum().rename("missing").to_frame()
    duplicates = int(df.duplicated().sum())
    desc = df[feature_numeric].describe().T
    skew = df[feature_numeric].skew().rename("skewness")
    target_summary = pd.DataFrame(
        {"count": target_counts, "percent": target_pct}
    ).rename(index=target_label)

    late_rates = {}
    for col in ["Product_importance", "Mode_of_Shipment", "Warehouse_block", "Gender", "Customer_care_calls"]:
        late_rates[col] = (df.groupby(col)[target].mean() * 100).round(2)

    # Strongest absolute correlations excluding self and duplicated pairs.
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            pairs.append((a, b, corr.loc[a, b], abs(corr.loc[a, b])))
    corr_pairs = sorted(pairs, key=lambda x: x[3], reverse=True)

    report = []
    report.append("# E-Commerce Shipping Data: Exploratory Data Analysis\n")
    report.append("## 1. Dataset Loading and Quality Checks\n")
    report.append(f"- Source file: `{DATA_PATH.name}`\n")
    report.append(f"- Shape: **{df.shape[0]:,} rows x {df.shape[1]:,} columns**\n")
    report.append(f"- Duplicate records: **{duplicates:,}**\n\n")
    report.append("### Column Data Types\n")
    report.append(dtypes.to_markdown())
    report.append("\n\n### Missing Values\n")
    report.append(missing.to_markdown())
    report.append("\n\n### Target Variable Distribution\n")
    report.append(target_summary.to_markdown())
    report.append("\n\n![Target class balance](figures/target_class_balance.png)\n")

    report.append("\n## 2. Univariate Analysis\n")
    report.append("### Numeric Feature Summary\n")
    numeric_summary = desc.join(skew).round(2)
    report.append(numeric_summary.to_markdown())
    report.append(
        "\n\nThe numeric distributions show a mix of compact count variables and wider continuous measures. "
        "`Customer_care_calls`, `Customer_rating`, and `Prior_purchases` are discrete with limited ranges. "
        "`Discount_offered` is strongly right-skewed, while `Cost_of_the_Product` and `Weight_in_gms` have broad spreads. "
        "The boxplots flag high values in `Discount_offered` and `Prior_purchases` as potential outliers.\n\n"
    )
    report.append("![Numeric univariate plots](figures/numeric_univariate.png)\n")
    report.append("\n### Categorical Feature Counts\n")
    for col in categorical_cols:
        counts = df[col].value_counts()
        report.append(f"\n**{col}**\n\n")
        report.append(counts.to_markdown())
        report.append("\n")
    report.append("\n![Categorical univariate plots](figures/categorical_univariate.png)\n")

    report.append("\n## 3. Bivariate Analysis Against Target\n")
    report.append("![Bivariate target relationships](figures/bivariate_target_relationships.png)\n\n")
    report.append(
        "`Discount_offered` separates the target classes clearly: shipments marked not on time tend to have larger discounts. "
        "`Weight_in_gms` also differs by class, with on-time shipments generally heavier in this dataset. "
        "Categorical late-rate plots show that `Product_importance`, `Mode_of_Shipment`, and `Customer_care_calls` have visible differences in not-on-time rate, "
        "although some gaps are moderate rather than dramatic.\n\n"
    )
    for col, rates in late_rates.items():
        report.append(f"**Not-on-time rate by {col} (%)**\n\n")
        report.append(rates.to_markdown())
        report.append("\n\n")

    report.append("## 4. Correlation Analysis\n")
    report.append("![Correlation heatmap](figures/numeric_correlation_heatmap.png)\n\n")
    report.append("The two strongest numeric relationships by absolute correlation are:\n")
    for a, b, value, _ in corr_pairs[:2]:
        report.append(f"- `{a}` vs `{b}`: **{value:.3f}**\n")
    report.append(
        "\nThe strongest positive relationship is between product cost and customer-care calls, suggesting higher-cost orders may attract more support interaction. "
        "The strongest target-related relationship is between `Discount_offered` and `Reached.on.Time_Y.N`, indicating discounts are closely connected to late-delivery labeling in this dataset.\n"
    )

    findings = (
        "The E-Commerce Shipping dataset contains 10,999 shipment records and 12 fields, with no missing values and no duplicate rows. "
        "The target variable is imbalanced but still usable for modelling: 6,563 shipments (59.67%) are labelled 1, meaning not delivered on time, while 4,436 shipments (40.33%) are labelled 0, meaning delivered on time. "
        "The imbalance is moderate, so evaluation should not rely on accuracy alone; precision, recall, F1-score and a confusion matrix would be more informative.\n\n"
        "The univariate plots show that several variables are discrete and bounded, including customer-care calls, customer rating and prior purchases. "
        "Customer ratings are fairly spread across the 1 to 5 scale, while prior purchases are concentrated at lower values with a few high observations. "
        "Discount offered is the most skewed numeric feature, with many small discounts and a smaller number of very high discounts. "
        "Cost of the product and shipment weight have wider spreads, and the boxplots suggest possible outliers in discount and prior purchases. "
        "Categorical fields appear clean: warehouse block F has the largest share, shipment mode is dominated by Ship, and product importance is mostly low or medium.\n\n"
        "The bivariate analysis suggests that Discount_offered is likely to be one of the most predictive features because not-on-time shipments have visibly higher discount values. "
        "Weight_in_gms also looks important, as the two target classes show different weight distributions, with on-time deliveries tending to be heavier. "
        "Customer_care_calls and Product_importance may add useful signal because their not-on-time rates vary by group, although their relationships are weaker than discount and weight. "
        "Mode_of_Shipment shows some variation, but the dominance of Ship means it should be interpreted carefully. "
        "The correlation heatmap supports these findings: Discount_offered has the strongest numeric association with the target, while Cost_of_the_Product and Customer_care_calls form the strongest non-target relationship. "
        "ID is a unique identifier and should be excluded from predictive modelling because it is not a meaningful shipment attribute."
    )
    word_count = len(findings.replace("\n", " ").split())
    report.append("\n## 5. Written Findings Report (300-400 words)\n")
    report.append(findings)
    report.append(f"\n\n_Word count: {word_count}_\n")

    out_path = OUT_DIR / "eda_report.md"
    out_path.write_text("".join(report), encoding="utf-8")

    # Save compact CSV summaries for auditability.
    numeric_summary.to_csv(OUT_DIR / "numeric_summary.csv")
    target_summary.to_csv(OUT_DIR / "target_summary.csv")
    corr.to_csv(OUT_DIR / "numeric_correlation_matrix.csv")
    print(f"Wrote {out_path}")
    print(f"Rows={df.shape[0]}, Cols={df.shape[1]}, Duplicates={duplicates}, Findings words={word_count}")


if __name__ == "__main__":
    main()
