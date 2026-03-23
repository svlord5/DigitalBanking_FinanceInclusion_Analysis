# ============================================================
#  DIGITAL BANKING, FINANCIAL INCLUSION & CO2 ANALYSIS
#  Run: python analysis.py
#  Colab: Upload and click Runtime > Run All
# ============================================================

import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUT = "/content" if os.path.exists("/content") else "figures"
os.makedirs(OUT, exist_ok=True)
print(f"Figures will be saved to: {OUT}/")
