# Fashion Retail Analytics: Reporting, Evaluation and Presentation (Week 6)

Final project of the Junior Data Science Analyst task series for the apparel and fashion industry.
It integrates data cleaning, exploratory analysis, demand forecasting and return-risk classification
into one evaluated, presentation-ready report.

> **Data note:** `data/raw_sales.csv` is a **simulated** weekly sales dataset (6 categories x 4 regions x
> 2 channels x 104 weeks) created by `src/generate_data.py` so the project runs end to end.
> Replace it with your own dataset from Weeks 1-5 to reproduce the analysis on real data.

## Key results (simulated data)

| Item | Result |
|---|---|
| Rows after cleaning | 4,992 (60 duplicates removed, 90 missing values imputed) |
| Best demand model | Random Forest, test R2 0.81, RMSE 12.2 units (baseline R2 -0.16) |
| Weekly aggregate forecast error | 8.4% WAPE |
| Return-risk classifier | Random Forest, F1 0.76, ROC-AUC 0.79 |
| Online vs store return rate | 17.2% vs 10.3% |

## Repository structure

```
fashion-retail-analytics/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw_sales.csv        # simulated raw data (with injected quality issues)
│   └── clean_sales.csv      # output of the cleaning step
├── src/
│   ├── generate_data.py     # creates the simulated dataset
│   └── analysis.py          # cleaning, EDA, models, evaluation, figures
├── figures/                 # 7 charts used in the report
└── reports/
    ├── Week6_Final_Report.docx
    └── metrics.json         # every number quoted in the report
```

## How to run

```bash
git clone https://github.com/<your-username>/fashion-retail-analytics.git
cd fashion-retail-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py     # skip if using your own data
python src/analysis.py
node src/build_report.js   # optional: rebuilds the .docx (needs Node and `npm i docx`)
```

## Methodology in brief

1. **Cleaning:** remove duplicates, standardise region labels, impute missing marketing spend with the category-channel median.
2. **EDA:** revenue mix, seasonality, discount and channel effects.
3. **Forecasting:** predict weekly units per category/region/channel. Time-based split (first 80 weeks train, last 24 test), compared against a mean baseline, with expanding-window cross-validation.
4. **Classification:** flag high return-rate rows (above the median).
5. **Evaluation:** MAE, RMSE, R2, WAPE for regression; accuracy, precision, recall, F1, ROC-AUC and confusion matrix for classification.

## Limitations

Simulated data, no external signals (weather, trends, competitor pricing), and cross-validation shows
instability on early folds that have not yet seen a full seasonal cycle.

## License

MIT
