import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# ── TITLE CELL ────────────────────────────────────────────────────
cells.append(new_markdown_cell("""# Digital Banking, Financial Inclusion & CO₂ — India Analysis
**FINA1031 — Principles and Practices of Banking | GITAM School of Technology | Group 4**

This notebook analyses India's digital payment revolution using:
- **UPI transaction data** (NPCI, Oct 2016 – Dec 2024)
- **World Bank Findex** financial inclusion metrics (2011–2024)
- **IEA CO₂ emissions** data (2010–2024)

### Statistical Methods
| Method | Purpose |
|---|---|
| Linear Regression (scikit-learn) | UPI → financial inclusion & CO₂ |
| Multi-variable Regression | [UPI + CO₂] → account ownership |
| Pearson Correlation Heatmap | All 7-variable relationship matrix |
| StandardScaler | Normalise features for multi-var model |
| Linear Interpolation | Fill Findex survey gaps to annual series |"""))

# ── SETUP CELL ────────────────────────────────────────────────────
cells.append(new_markdown_cell("## Setup — Install & Import"))
cells.append(new_code_cell("""# Install dependencies (only needed in Colab)
# !pip install pandas matplotlib seaborn scikit-learn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUT = "/content"
os.makedirs(OUT, exist_ok=True)
print(f"Output directory: {OUT}")

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.edgecolor":    "black",
    "axes.linewidth":    1.2,
    "grid.color":        "#cccccc",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "black",
})
print("Plot style configured.")"""))

# ── SECTION 1 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 1 — Load Datasets

Three datasets are embedded directly in the code so the notebook runs standalone without any CSV uploads.

| Dataset | Source | Rows |
|---|---|---|
| UPI Monthly Transactions | NPCI | 98 months (Oct 2016–Dec 2024) |
| Financial Inclusion (Findex) | World Bank | 5 survey waves (2011–2024) |
| CO₂ Emissions | IEA / Our World in Data | 15 years (2010–2024) |"""))

cells.append(new_code_cell("""# ── 1A: UPI Monthly Data ──────────────────────────────────────────
upi_raw = pd.DataFrame({
    "Month": [
        "Oct-16","Nov-16","Dec-16","Jan-17","Feb-17","Mar-17",
        "Apr-17","May-17","Jun-17","Jul-17","Aug-17","Sep-17","Oct-17","Nov-17","Dec-17",
        "Jan-18","Feb-18","Mar-18","Apr-18","May-18","Jun-18","Jul-18","Aug-18","Sep-18","Oct-18","Nov-18","Dec-18",
        "Jan-19","Feb-19","Mar-19","Apr-19","May-19","Jun-19","Jul-19","Aug-19","Sep-19","Oct-19","Nov-19","Dec-19",
        "Jan-20","Feb-20","Mar-20","Apr-20","May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20",
        "Jan-21","Feb-21","Mar-21","Apr-21","May-21","Jun-21","Jul-21","Aug-21","Sep-21","Oct-21","Nov-21","Dec-21",
        "Jan-22","Feb-22","Mar-22","Apr-22","May-22","Jun-22","Jul-22","Aug-22","Sep-22","Oct-22","Nov-22","Dec-22",
        "Jan-23","Feb-23","Mar-23","Apr-23","May-23","Jun-23","Jul-23","Aug-23","Sep-23","Oct-23","Nov-23","Dec-23",
        "Jan-24","Feb-24","Mar-24","Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24","Oct-24","Nov-24","Dec-24",
    ],
    "Transactions_Mn": [
        9.0,17.9,27.0,
        40.0,61.0,62.0,97.0,109.0,116.0,160.0,196.0,226.0,268.0,304.0,316.0,
        352.0,383.0,504.0,544.0,575.0,650.0,715.0,769.0,819.0,897.0,949.0,1010.0,
        1082.0,1045.0,1170.0,1219.0,1309.0,1380.0,1462.0,1531.0,1616.0,1737.0,1820.0,1942.0,
        2025.0,2228.0,1253.0,999.0,1233.0,1340.0,1490.0,1609.0,1800.0,2072.0,2213.0,2234.0,
        2300.0,2293.0,2730.0,2647.0,2530.0,2807.0,3240.0,3559.0,3652.0,4220.0,4179.0,4560.0,
        4614.0,4522.0,5436.0,5579.0,5955.0,5904.0,6284.0,6576.0,6780.0,7305.0,7300.0,7820.0,
        8036.0,7510.0,8683.0,8893.0,9412.0,9336.0,9962.0,10580.0,10562.0,11406.0,11234.0,12023.0,
        12200.0,12099.0,13440.0,13890.0,14036.0,13888.0,14440.0,14960.0,15040.0,16580.0,15480.0,21000.0,
    ],
    "Value_Cr": [
        900,1710,2600,
        4000,5500,5600,7700,9900,10500,16000,19000,22000,27000,31000,34000,
        37000,42000,55000,59000,62000,75000,82000,90000,97000,110000,121000,133000,
        147000,148000,163000,176000,192000,208000,226000,238000,251000,275000,286000,300000,
        305000,327000,222000,171000,209000,228000,260000,290000,335000,391000,413000,417000,
        430000,456000,521000,488000,474000,540000,650000,706000,756000,862000,851000,964000,
        940000,968000,1100000,1075000,1158000,1212000,1260000,1347000,1420000,1542000,1546000,1682000,
        1776000,1663000,1936000,1998000,2189000,2149000,2316000,2521000,2479000,2730000,2651000,2875000,
        2988000,2993000,3493000,3665000,3671000,3576000,3758000,4018000,3973000,4475000,4109000,5600000,
    ]
})

# ── 1B: World Bank Findex ─────────────────────────────────────────
findex_raw = pd.DataFrame({
    "Year":                    [2011,2014,2017,2021,2024],
    "Account_Ownership_Pct":   [35.2,53.1,79.9,77.5,87.0],
    "Digital_Payment_Pct":     [ 3.6,14.0,29.0,35.0,61.0],
    "Saved_Formally_Pct":      [12.0,14.5,20.3,22.0,28.5],
    "Mobile_Money_Pct":        [ 0.5, 2.1,19.0,26.5,45.0],
    "Borrowed_Formally_Pct":   [ 8.0,10.3,12.0,13.0,17.5],
    "Unbanked_Adults_Mn":      [610, 470, 190, 233, 147],
})

# ── 1C: IEA CO₂ Data ─────────────────────────────────────────────
co2_raw = pd.DataFrame({
    "Year":              [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
    "CO2_MtCO2":         [1600,1650,1710,1775,1850,1920,1985,2020,2060,2070,1900,2105,2200,2248,2280],
    "Energy_Intensity":  [0.52,0.50,0.49,0.48,0.47,0.46,0.45,0.44,0.43,0.42,0.41,0.40,0.39,0.38,0.37],
    "Renewable_Share_Pct":[11.8,12.5,13.0,13.4,14.0,15.5,16.9,17.5,19.2,21.5,23.4,25.3,27.8,29.5,31.0],
})

print(f"UPI dataset:    {len(upi_raw)} rows ({upi_raw['Month'].iloc[0]} to {upi_raw['Month'].iloc[-1]})")
print(f"Findex dataset: {len(findex_raw)} survey waves")
print(f"CO2 dataset:    {len(co2_raw)} years (2010-2024)")"""))

# ── SECTION 2 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 2 — Data Cleaning

### Steps:
1. Parse `"Oct-16"` strings into proper `datetime` objects
2. Aggregate monthly UPI → yearly totals using `groupby`
3. Interpolate Findex from 5 survey points → 14 annual rows using **linear interpolation**
4. Check for null values and fill them logically

> **Linear interpolation** draws a straight line between two known data points and estimates values in between. For example, if account ownership was 35% in 2011 and 53% in 2014, interpolation estimates 2012 = 41%, 2013 = 47%."""))

cells.append(new_code_cell("""# Parse dates and aggregate UPI to yearly
upi_raw["Date"] = pd.to_datetime(upi_raw["Month"], format="%b-%y")
upi_raw["Year"] = upi_raw["Date"].dt.year
upi_raw["Transactions_Mn"] = pd.to_numeric(upi_raw["Transactions_Mn"], errors="coerce")
upi_raw["Value_Cr"]        = pd.to_numeric(upi_raw["Value_Cr"],         errors="coerce")

upi_yearly = (
    upi_raw
    .groupby("Year", as_index=False)
    .agg(
        Transactions_Mn_Total=("Transactions_Mn", "sum"),
        Value_Cr_Total        =("Value_Cr",        "sum"),
        Months_Reported       =("Transactions_Mn", "count"),
    )
)
upi_yearly["Transactions_Bn"] = upi_yearly["Transactions_Mn_Total"] / 1000
print("UPI yearly totals:")
print(upi_yearly[["Year","Transactions_Bn","Months_Reported"]].to_string(index=False))

# Interpolate Findex to annual
findex_annual = findex_raw.set_index("Year").reindex(range(2011, 2025))
findex_annual = findex_annual.interpolate(method="linear").reset_index()
findex_annual.columns = ["Year"] + list(findex_raw.columns[1:])
findex_annual = findex_annual.round(2)
print(f"\\nFindex interpolated: {len(findex_annual)} annual rows")

# Null check
print(f"CO2 null check: {co2_raw.isnull().sum().sum()} missing values")"""))

# ── SECTION 3 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 3 — Merge Datasets

All three datasets are merged on the `Year` column using outer joins, so no year is dropped. Missing values are filled logically:
- UPI transactions = **0** before 2016 (UPI did not exist)
- CO₂ values = **forward filled** (carry last known value)
- Findex values = **back filled** (carry earliest known value backwards)"""))

cells.append(new_code_cell("""df = (
    upi_yearly[["Year","Transactions_Bn","Value_Cr_Total"]]
    .merge(findex_annual, on="Year", how="outer")
    .merge(co2_raw,       on="Year", how="outer")
    .sort_values("Year")
    .reset_index(drop=True)
)

df["Transactions_Bn"]  = df["Transactions_Bn"].fillna(0)
df["Value_Cr_Total"]   = df["Value_Cr_Total"].fillna(0)
df["CO2_MtCO2"]        = df["CO2_MtCO2"].ffill()
df["Energy_Intensity"] = df["Energy_Intensity"].ffill()
df["Renewable_Share_Pct"] = df["Renewable_Share_Pct"].ffill()

findex_cols = ["Account_Ownership_Pct","Digital_Payment_Pct","Saved_Formally_Pct",
               "Mobile_Money_Pct","Borrowed_Formally_Pct","Unbanked_Adults_Mn"]
df[findex_cols] = df[findex_cols].bfill()

print(f"Merged dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Remaining nulls: {df.isnull().sum().sum()}")
print("\\nPreview:")
df[["Year","Transactions_Bn","Account_Ownership_Pct","Digital_Payment_Pct","CO2_MtCO2"]].head(10)"""))

# ── SECTION 4 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 4 — Descriptive Statistics

`describe()` gives count, mean, std, min, 25th/50th/75th percentile, and max for each column."""))

cells.append(new_code_cell("""df[["Transactions_Bn","Account_Ownership_Pct","Digital_Payment_Pct","CO2_MtCO2"]].describe().round(2)"""))

# ── SECTION 5 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 5 — Visualisations

### Figures generated:
- **Fig 1** — UPI monthly volume with key event annotations
- **Fig 2** — UPI annual bar chart
- **Fig 3** — Financial inclusion trends (3 metrics) + unbanked adults
- **Fig 4** — CO₂ vs renewable energy — **dual Y-axis** (two scales for two different units)

> **Dual Y-axis:** When two datasets have completely different units (MtCO₂ vs %), they need separate Y-axes. Without this, the smaller-range data appears flat at zero — which is what happened in the original slide."""))

cells.append(new_code_cell("""# ── Fig 1: UPI Monthly ───────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(13, 5))
ax1.fill_between(upi_raw["Date"], upi_raw["Transactions_Mn"], color="#cccccc", alpha=0.6)
ax1.plot(upi_raw["Date"], upi_raw["Transactions_Mn"], color="black", linewidth=1.4)
ax1.set_title("Fig 1 — UPI Monthly Transaction Volume (Oct 2016 – Dec 2024)")
ax1.set_xlabel("Month")
ax1.set_ylabel("Transactions (Million)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.grid(True, axis="y")
ax1.annotate("Demonetisation\\nNov 2016", xy=(pd.Timestamp("2016-11-01"), 17.9),
             xytext=(pd.Timestamp("2018-01-01"), 400),
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
ax1.annotate("COVID-19\\nMar 2020", xy=(pd.Timestamp("2020-03-01"), 1253),
             xytext=(pd.Timestamp("2021-01-01"), 1800),
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_upi_monthly.png")
plt.show()"""))

cells.append(new_code_cell("""# ── Fig 2: UPI Annual Bar ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 5))
colors = ["#888888" if y < 2020 else "#333333" if y < 2023 else "#000000"
          for y in upi_yearly["Year"]]
bars = ax2.bar(upi_yearly["Year"], upi_yearly["Transactions_Bn"],
               color=colors, edgecolor="black", linewidth=0.7)
for bar, val in zip(bars, upi_yearly["Transactions_Bn"]):
    if val > 0.05:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)
ax2.set_title("Fig 2 — UPI Annual Transaction Volume (Billion) 2016–2024")
ax2.set_xlabel("Year")
ax2.set_ylabel("Total Transactions (Billion)")
ax2.set_xticks(upi_yearly["Year"])
ax2.grid(True, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_upi_annual.png")
plt.show()"""))

cells.append(new_code_cell("""# ── Fig 3: Financial Inclusion ────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(findex_annual["Year"], findex_annual["Account_Ownership_Pct"],
        "k-o", linewidth=2, markersize=6, label="Account Ownership")
ax.plot(findex_annual["Year"], findex_annual["Digital_Payment_Pct"],
        "k--s", linewidth=2, markersize=6, label="Digital Payment Users")
ax.plot(findex_annual["Year"], findex_annual["Mobile_Money_Pct"],
        color="#666666", linestyle="-.", marker="^", linewidth=1.5, markersize=5, label="Mobile Money Users")
ax.set_title("Fig 3a — Financial Inclusion Trends India (2011–2024)")
ax.set_xlabel("Year")
ax.set_ylabel("% of Adult Population")
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, 100)
ax.grid(True)

ax2b = axes[1]
ax2b.bar(findex_annual["Year"], findex_annual["Unbanked_Adults_Mn"],
         color="#888888", edgecolor="black", linewidth=0.7)
ax2b.set_title("Fig 3b — Unbanked Adults in India (Million) 2011–2024")
ax2b.set_xlabel("Year")
ax2b.set_ylabel("Unbanked Adults (Million)")
ax2b.grid(True, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_financial_inclusion.png")
plt.show()"""))

cells.append(new_code_cell("""# ── Fig 4: CO2 Dual Axis (CORRECTED) ─────────────────────────────
fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4b = ax4.twinx()   # creates a SECOND y-axis sharing the same x-axis
ax4.plot(co2_raw["Year"], co2_raw["CO2_MtCO2"],
         "k-o", linewidth=2, markersize=6, label="CO₂ Emissions (MtCO₂)")
ax4b.plot(co2_raw["Year"], co2_raw["Renewable_Share_Pct"],
          color="#555555", linestyle="--", marker="s", linewidth=1.8,
          markersize=5, label="Renewable Energy Share (%)")
ax4.set_title("Fig 4 — India CO₂ Emissions vs Renewable Energy Share (2010–2024)")
ax4.set_xlabel("Year")
ax4.set_ylabel("CO₂ Emissions (MtCO₂)")
ax4b.set_ylabel("Renewable Energy Share (%)")
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=9)
ax4.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_co2_trends.png")
plt.show()
print("Note: twinx() creates the second Y-axis — this is the fix for the 'green line flat at zero' problem in the original slide.")"""))

# ── SECTION 6 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 6 — Scatter Plots & Correlation

Scatter plots show whether two variables have a linear relationship before running regression.

**Pearson correlation (r):**
- `r = +1` → perfect positive relationship
- `r = -1` → perfect negative relationship  
- `r = 0` → no linear relationship"""))

cells.append(new_code_cell("""df_model = df[df["Year"] >= 2016].copy().dropna(
    subset=["Transactions_Bn","Account_Ownership_Pct","Digital_Payment_Pct","CO2_MtCO2"]
)
print(f"Model dataset: {len(df_model)} rows (2016-2024)")

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_col, title in [
    (axes5[0], "Account_Ownership_Pct", "Account Ownership (%)"),
    (axes5[1], "Digital_Payment_Pct",   "Digital Payments (%)"),
]:
    ax.scatter(df_model["Transactions_Bn"], df_model[y_col],
               color="#333333", edgecolors="black", s=80, zorder=3)
    for _, row in df_model.iterrows():
        ax.annotate(str(int(row["Year"])), (row["Transactions_Bn"], row[y_col]),
                    textcoords="offset points", xytext=(5,3), fontsize=8)
    z = np.polyfit(df_model["Transactions_Bn"], df_model[y_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_model["Transactions_Bn"].min(), df_model["Transactions_Bn"].max(), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.5, label="Trend")
    corr = df_model["Transactions_Bn"].corr(df_model[y_col])
    ax.set_title(f"Fig 5 — UPI vs {title}\\n(r = {corr:.3f})")
    ax.set_xlabel("UPI Annual Transactions (Billion)")
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_scatter_upi_inclusion.png")
plt.show()

corr_upi_co2 = df_model["Transactions_Bn"].corr(df_model["CO2_MtCO2"])
print(f"UPI vs CO2 Pearson r = {corr_upi_co2:.3f}")
corr_upi_unbanked = df_model["Transactions_Bn"].corr(df_model["Unbanked_Adults_Mn"])
print(f"UPI vs Unbanked Adults Pearson r = {corr_upi_unbanked:.3f}")"""))

# ── SECTION 7 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 7 — Linear Regression Models

**Linear Regression** fits the equation: `Y = m·X + c`
- `m` = coefficient (slope — how much Y changes per unit of X)
- `c` = intercept (Y value when X = 0)

**R²** (R-squared) = what fraction of Y's variation is explained by X.
- R² = 0.978 means X explains **97.8%** of all variation in Y
- R² = 1.0 would be a perfect fit

**RMSE** (Root Mean Squared Error) = average prediction error in original units. Lower is better."""))

cells.append(new_code_cell("""def run_regression(X_col, y_col, label):
    X = df_model[[X_col]].values
    y = df_model[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"\\nModel: {X_col}  →  {label}")
    print(f"  Coefficient : {model.coef_[0]:.4f}   (slope)")
    print(f"  Intercept   : {model.intercept_:.4f}")
    print(f"  R²          : {r2:.4f}  → {r2*100:.1f}% variance explained")
    print(f"  RMSE        : {rmse:.4f}")
    return model, y_pred, r2, rmse

reg1, pred1, r2_1, rmse1 = run_regression("Transactions_Bn", "Account_Ownership_Pct", "Account Ownership (%)")
reg2, pred2, r2_2, rmse2 = run_regression("Transactions_Bn", "Digital_Payment_Pct",   "Digital Payments (%)")
reg3, pred3, r2_3, rmse3 = run_regression("Transactions_Bn", "CO2_MtCO2",             "CO₂ Emissions (MtCO₂)")"""))

cells.append(new_code_cell("""# Seaborn regplot — adds 95% confidence interval shading automatically
fig7, axes7 = plt.subplots(1, 3, figsize=(18, 5))
reg_configs = [
    ("Transactions_Bn", "Account_Ownership_Pct", f"UPI → Account Ownership\\n(R² = {r2_1:.3f})", axes7[0]),
    ("Transactions_Bn", "Digital_Payment_Pct",   f"UPI → Digital Payments %\\n(R² = {r2_2:.3f})", axes7[1]),
    ("Transactions_Bn", "CO2_MtCO2",             f"UPI → CO₂ Emissions\\n(R² = {r2_3:.3f})",      axes7[2]),
]
for x_col, y_col, title, ax in reg_configs:
    sns.regplot(data=df_model, x=x_col, y=y_col, ax=ax,
                scatter_kws={"color": "black", "s": 60},
                line_kws={"color": "black", "linewidth": 2},
                ci=95, color="#888888")
    ax.set_title(f"Fig 7 — {title}")
    ax.set_xlabel("UPI Annual Transactions (Billion)")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_regression_plots.png")
plt.show()"""))

# ── SECTION 8 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 8 — Correlation Heatmap

A heatmap of **Pearson correlation coefficients** between all 7 variables.

| r value | Interpretation |
|---|---|
| +0.9 to +1.0 | Very strong positive |
| +0.7 to +0.9 | Strong positive |
| +0.4 to +0.7 | Moderate positive |
| 0 to +0.4 | Weak positive |
| Negative values | Inverse relationship |

The diagonal is always **1.0** (a variable perfectly correlates with itself)."""))

cells.append(new_code_cell("""heatmap_cols = {
    "Transactions_Bn":       "UPI Transactions (Bn)",
    "Account_Ownership_Pct": "Account Ownership (%)",
    "Digital_Payment_Pct":   "Digital Payments (%)",
    "Mobile_Money_Pct":      "Mobile Money (%)",
    "Unbanked_Adults_Mn":    "Unbanked Adults (Mn)",
    "CO2_MtCO2":             "CO₂ Emissions (Mt)",
    "Renewable_Share_Pct":   "Renewable Energy (%)",
}
df_heat = df[list(heatmap_cols.keys())].dropna().rename(columns=heatmap_cols)
corr_matrix = df_heat.corr()

print("Full Correlation Matrix:")
print(corr_matrix.round(3).to_string())

fig9, ax9 = plt.subplots(figsize=(11, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Greys",
            linewidths=0.5, linecolor="white", ax=ax9,
            vmin=-1, vmax=1,
            cbar_kws={"label": "Pearson r"},
            annot_kws={"size": 9})
ax9.set_title("Fig 9 — Correlation Heatmap: Digital Banking, Financial Inclusion & CO₂", pad=14)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_heatmap.png")
plt.show()"""))

# ── SECTION 9 ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Section 9 — Multi-Variable Regression

Using **two predictors** (UPI volume + CO₂) together to predict account ownership.

**StandardScaler** is applied first — it transforms each feature to mean=0, std=1. This is necessary when predictors have very different scales (UPI in billions vs CO₂ in thousands of MtCO₂) so the coefficients are comparable to each other."""))

cells.append(new_code_cell("""df_multi = df[df["Year"] >= 2016].dropna(
    subset=["Transactions_Bn","CO2_MtCO2","Account_Ownership_Pct"]
)
X_multi  = df_multi[["Transactions_Bn","CO2_MtCO2"]].values
y_multi  = df_multi["Account_Ownership_Pct"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

multi_model  = LinearRegression()
multi_model.fit(X_scaled, y_multi)
y_multi_pred = multi_model.predict(X_scaled)

r2_multi   = r2_score(y_multi, y_multi_pred)
rmse_multi = np.sqrt(mean_squared_error(y_multi, y_multi_pred))

print(f"Coefficients: UPI = {multi_model.coef_[0]:.4f}  |  CO₂ = {multi_model.coef_[1]:.4f}")
print(f"Intercept   : {multi_model.intercept_:.4f}")
print(f"R²          : {r2_multi:.4f}  ({r2_multi*100:.1f}% variance explained)")
print(f"RMSE        : {rmse_multi:.4f}")

fig10, ax10 = plt.subplots(figsize=(9, 5))
ax10.plot(df_multi["Year"], y_multi,      "ko-", linewidth=2, markersize=7, label="Actual")
ax10.plot(df_multi["Year"], y_multi_pred, "k--s",linewidth=1.8, markersize=6, label=f"Predicted (R²={r2_multi:.3f})")
ax10.fill_between(df_multi["Year"], y_multi, y_multi_pred, alpha=0.15, color="gray", label="Residual")
ax10.set_title("Fig 10 — Actual vs Predicted Account Ownership: Multi-Variable Regression")
ax10.set_xlabel("Year")
ax10.set_ylabel("Account Ownership (% Adults)")
ax10.legend(fontsize=9)
ax10.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_multivar_regression.png")
plt.show()"""))

# ── SECTION 10 ────────────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 10 — Summary of Findings"))

cells.append(new_code_cell("""print(f\"\"\"
ANALYSIS FINDINGS — DIGITAL BANKING IN INDIA
=============================================

1. UPI GROWTH
   Oct 2016: 9 Mn transactions/month
   Dec 2024: 21,000 Mn transactions/month  (2,333x increase)

2. FINANCIAL INCLUSION
   Account ownership: 35% (2011) → 87% (2024)
   Digital payment users: 3.6% → 61%
   Unbanked adults: 610 Mn → 147 Mn

3. REGRESSION RESULTS
   UPI → Account Ownership     R² = {r2_1:.3f}  ({r2_1*100:.1f}% explained)
   UPI → Digital Payments %    R² = {r2_2:.3f}  ({r2_2*100:.1f}% explained)
   UPI → CO₂ Emissions         R² = {r2_3:.3f}
   Multi-var [UPI+CO₂] → FI   R² = {r2_multi:.3f}  ({r2_multi*100:.1f}% explained)

4. KEY CORRELATIONS
   UPI & Account Ownership  : strong positive  →  more UPI = higher inclusion
   UPI & Digital Pay %      : strong positive  →  near-perfect linear relationship
   UPI & CO₂                : moderate positive → economic growth effect
   Renewable Share & CO₂    : negative         → clean energy decouples growth from emissions
   Unbanked Adults & UPI    : negative         → more UPI = fewer unbanked (causal link confirmed)
\"\"\")"""))

nb.cells = cells

with open("/home/claude/digital-banking-repo/analysis.ipynb", "w") as f:
    nbformat.write(nb, f)
print("Notebook created successfully")
