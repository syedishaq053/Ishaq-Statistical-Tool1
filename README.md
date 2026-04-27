# 📊 Ishaq Statistical Analysis Tool

A comprehensive web-based statistical analysis application built with Streamlit.

## Features

- **Multiple statistical tests**: t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis
- **Paired and independent comparisons**
- **Effect size calculations**: Cohen's d, eta-squared, rank-biserial correlation
- **Post-hoc analyses**: Tukey HSD, Dunn's test with Bonferroni correction
- **Correlation analysis**: Pearson and Spearman with heatmap visualization
- **Boxplots with significance annotations**
- **Bland-Altman plots** for method comparison
- **Outlier detection** using IQR method
- **Export results** as CSV and PNG

## Installation

### Local Development

1. Clone this repository:
   ```bash
   git clone [github.com](https://github.com/syedishaq053/ishaq-stats-app.git)
   cd ishaq-stats-app

2. Create a virtual environment

PowerShell (recommended on Windows):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

Command Prompt (cmd.exe):

```cmd
python -m venv venv
venv\Scripts\activate
```

Git Bash / WSL (Unix-style):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application 

streamlit run app.py

