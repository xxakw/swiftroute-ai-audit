# SwiftRoute AI Practical Assessment

This repository contains the Part 1 exploratory data analysis for the E-Commerce Shipping Data dataset used as SwiftRoute shipment records.

## Topics Covered

- ML paradigms and exploratory preparation for supervised learning
- Logic and knowledge foundations for later rule-based analysis
- Search and CSP concepts for shipment-routing reasoning
- AI foundations and practical data-quality assessment

## Repository Structure

- `data/Train.csv` - working dataset downloaded from Kaggle
- `src/eda_shipping.py` - reproducible pandas EDA script
- `reports/eda_report.md` - detailed markdown EDA report
- `reports/eda_report.pdf` - submission PDF report
- `reports/figures/` - generated EDA charts
- `summaries/` - exported numeric summaries and correlation matrix

## Dataset

Dataset: E-Commerce Shipping Data by prachi13 on Kaggle  
URL: https://www.kaggle.com/datasets/prachi13/customer-analytics

Target field: `Reached.on.Time_Y.N`, where `1` means not delivered on time and `0` means delivered on time.

## Reproduce the Analysis

```bash
pip install pandas matplotlib seaborn tabulate
python src/eda_shipping.py
```

The script loads `data/Train.csv`, performs quality checks, creates univariate and bivariate plots, generates a numeric correlation heatmap, and writes the report outputs.

## Submission Note

The current package fully covers the provided Part 1 EDA requirements. Add later assessment parts to this repository when their complete instructions are available.
