from footpredict.config import get_settings
from footpredict.data.storage import read_parquet, write_parquet
from footpredict.features.pipeline import prepare_training_frame
from footpredict.models.backtest import walk_forward_backtest
from footpredict.models.train import train_model, FEATURES
from footpredict.models.registry import save_model

import pandas as pd


def main():
    s = get_settings()

    processed = list((s.data_dir / "processed").glob("matches_t*.parquet"))
    if not processed:
        raise RuntimeError("Aucun dataset processed trouvé. Lance scripts/build_dataset.py d'abord.")
    path = sorted(processed)[-1]

    df = read_parquet(path)
    df_ml = prepare_training_frame(df, form_window=5)

    if df_ml is None or df_ml.empty:
        raise RuntimeError("df_ml est vide/None. Vérifie prepare_training_frame().")

    print("DF_ML columns:", list(df_ml.columns))
    print("DF_ML shape:", df_ml.shape)

    metrics = walk_forward_backtest(df_ml, n_splits=5)
    print("Backtest:", {k: v for k, v in metrics.items() if k != "oof_proba"})

    model = train_model(df_ml)
    save_model((s.data_dir / "reports" / "model.joblib"), model)

    # -------------------------
    # Feature importance (LogReg coef)
    # -------------------------
    # ton pipeline = ("scaler", ...), ("clf", LogisticRegression)
    coef = model.named_steps["clf"].coef_[0]

    df_imp = pd.DataFrame({"feature": FEATURES, "importance": coef})
    df_imp["abs_importance"] = df_imp["importance"].abs()
    df_imp = df_imp.sort_values("abs_importance", ascending=False)

    df_imp.to_csv(s.data_dir / "reports" / "feature_importance.csv", index=False)

    print("\nFeature importance (top 20):")
    print(df_imp.head(20))

    # -------------------------
    # Save training frame
    # -------------------------
    df_ml = df_ml.copy()
    df_ml["oof_proba"] = metrics["oof_proba"]
    write_parquet((s.data_dir / "reports" / "train_frame.parquet"), df_ml)

    print("✅ saved model + reports in data/reports/")


if __name__ == "__main__":
    main()