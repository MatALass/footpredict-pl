import pandas as pd

df = pd.read_parquet("data/reports/train_frame.parquet").copy()
target = df["y_homewin"].astype(int)

# Exclure leakage / identifiants / post-match
drop_cols = {
    "y_homewin",
    "home_goals",
    "away_goals",
    "event_id",
    "timestamp",
    "round",
}

num = df.select_dtypes(include="number").drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

corr = num.apply(lambda s: s.corr(target)).sort_values(ascending=False)

print(corr.head(30))
corr.to_csv("data/reports/feature_correlation.csv")