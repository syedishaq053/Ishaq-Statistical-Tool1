# Ishaq Statistical Analysis Suite

A comprehensive web‑based statistical analysis platform with two
integrated tools:

1.  **Statistical Analysis Tool** -- full parametric & non‑parametric
    tests, effect sizes, post‑hoc, regression, correlation, and
    publication‑ready plots.
2.  **Sample Size Calculator** -- power analysis directly from your data
    or manually entered parameters (G\*Power‑validated).

**Live demo**: https://www.meddatastats.com/
*Free for Indian users; 90‑day trial for others.*

------------------------------------------------------------------------

## 📊 Features

### Statistical Analysis Tool

-   Upload Excel/CSV, handle missing data, outlier removal
-   Parametric & non‑parametric tests
-   Effect sizes and post‑hoc analysis
-   Correlation matrices and visualizations
-   Export results as CSV, PNG, PDF

### Sample Size Calculator

-   A priori, post hoc, sensitivity analysis
-   Multiple statistical test families
-   Interactive power curves
-   Downloadable reports

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Streamlit
-   SciPy, Statsmodels, Scikit-learn
-   Matplotlib, Seaborn
-   Supabase (PostgreSQL)
-   Render + Cloudflare

------------------------------------------------------------------------

## 🚀 Setup

``` bash
git clone https://github.com/your-username/statistical-analysis-suite.git
cd statistical-analysis-suite
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 🔐 Secrets Configuration

Create `.streamlit/secrets.toml`:

``` toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

------------------------------------------------------------------------

## 📄 License

MIT License


