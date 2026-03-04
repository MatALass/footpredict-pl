import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Optional (metrics + plots)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

# ---------------------------
# Page config + small CSS
# ---------------------------
st.set_page_config(page_title="FootPredict", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .big-kpi {font-size: 2.2rem; font-weight: 800;}
      .muted {color: #6b7280;}
      .card {border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px; background: #ffffff;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; border:1px solid #e5e7eb;}
      .pill-green {background:#ecfdf5; border-color:#a7f3d0;}
      .pill-yellow{background:#fffbeb; border-color:#fde68a;}
      .pill-red{background:#fef2f2; border-color:#fecaca;}
      .hr {height:1px; background:#e5e7eb; margin: 10px 0 18px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Paths
# ---------------------------
DATA_DIR = Path("data")
REPORTS = DATA_DIR / "reports"

TRAIN_PATH = REPORTS / "train_frame.parquet"
NEXT_PATH = REPORTS / "predictions_next.parquet"
IMP_PATH = REPORTS / "feature_importance.csv"
CORR_PATH = REPORTS / "feature_correlation.csv"
MODEL_PATH = REPORTS / "model.joblib"

# ---------------------------
# Helpers
# ---------------------------
def need(path: Path, help_msg: str):
    if not path.exists():
        st.warning(help_msg)
        st.stop()

def fmt_pct(x):
    if x is None or pd.isna(x):
        return "-"
    return f"{100*x:.1f}%"

def conf_label(p: float):
    if p >= 0.65:
        return ("Forte", "pill pill-green")
    if p >= 0.55:
        return ("Moyenne", "pill pill-yellow")
    return ("Faible", "pill pill-red")

def safe_read_corr(path: Path):
    # CSV from a Series => often 2 columns without header
    try:
        df = pd.read_csv(path, header=None, names=["feature", "corr"]).dropna()
        df = df[df["feature"] != "feature"]
        return df
    except Exception:
        df = pd.read_csv(path, index_col=0).reset_index()
        df.columns = ["feature", "corr"]
        return df

def style_proba_table(df: pd.DataFrame, proba_col: str):
    # simple gradient on probability column
    return df.style.format({proba_col: "{:.3f}"}).background_gradient(subset=[proba_col])

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="card">
          <div class="muted">{title}</div>
          <div class="big-kpi">{value}</div>
          <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Load data
# ---------------------------
need(TRAIN_PATH, "⚠️ Lance `python scripts/train_backtest.py` pour générer `data/reports/train_frame.parquet`.")
df_train = pd.read_parquet(TRAIN_PATH).copy()
df_train["date"] = pd.to_datetime(df_train["date"], errors="coerce")

df_next = None
if NEXT_PATH.exists():
    df_next = pd.read_parquet(NEXT_PATH).copy()
    if "date" in df_next.columns:
        df_next["date"] = pd.to_datetime(df_next["date"], errors="coerce")

df_imp = pd.read_csv(IMP_PATH) if IMP_PATH.exists() else None
df_corr = safe_read_corr(CORR_PATH) if CORR_PATH.exists() else None

teams = sorted(set(df_train["home"]).union(set(df_train["away"])))

# ---------------------------
# Sidebar nav
# ---------------------------
st.title("⚽ FootPredict — Premier League")
st.caption("Modèle ML (LogReg + Elo + Form) • Probabilités Home Win • Analyse & prédictions")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📅 Next Matches", "📈 Performance", "🔎 Match Explainer", "🧠 Model Insights", "ℹ️ Glossary"],
)
st.sidebar.markdown("---")
team_filter = st.sidebar.selectbox("Filtrer par équipe", ["(toutes)"] + teams)

# If you use run_all.ps1
with st.sidebar.expander("▶️ Lancer tout (run_all.ps1)"):
    st.code(r"powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1")

# ---------------------------
# Precompute metrics from OOF (history)
# ---------------------------
has_oof = "oof_proba" in df_train.columns and df_train["oof_proba"].notna().any()
metrics = {}
if has_oof:
    y_true = df_train["y_homewin"].astype(int).values
    y_prob = df_train["oof_proba"].clip(1e-6, 1 - 1e-6).values
    y_pred = (y_prob >= 0.5).astype(int)

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["logloss"] = float(log_loss(y_true, y_prob))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)
else:
    cm = None

# ---------------------------
# Page: Dashboard
# ---------------------------
if page == "🏠 Dashboard":
    st.markdown("## Résumé rapide")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Matchs (train)", f"{len(df_train)}", "Matchs terminés utilisés pour l’entraînement")
    with c2:
        # Important: this is NOT accuracy, it's mean predicted probability
        avg_p = float(df_train["oof_proba"].mean()) if has_oof else None
        kpi_card("Proba moyenne (OOF)", fmt_pct(avg_p), "Moyenne des probabilités Home Win (≠ précision)")
    with c3:
        kpi_card("Modèle", "✅ model.joblib" if MODEL_PATH.exists() else "❌ absent", "Fichier du modèle entraîné")
    with c4:
        kpi_card("Prédictions", "✅ next.parquet" if df_next is not None else "❌ absentes", "Fichier des prochains matchs")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.info(
        "**Important :** `Proba moyenne (OOF)` n’est **pas** la précision.\n\n"
        f"- Ta vraie performance (d’après l’OOF) est plutôt autour de : **accuracy ≈ {fmt_pct(metrics.get('accuracy'))}**.\n"
        "- Une proba moyenne à 36% veut juste dire : le modèle donne en moyenne 0.36 de chance de victoire à domicile."
    )

    view = df_train.copy()
    if team_filter != "(toutes)":
        view = view[(view["home"] == team_filter) | (view["away"] == team_filter)]

    st.markdown("### Historique (OOF) — derniers matchs")
    cols = ["date", "home", "away", "home_goals", "away_goals", "y_homewin", "oof_proba"]
    cols = [c for c in cols if c in view.columns]
    hist = view.sort_values("date").tail(40)[cols].copy()
    hist.rename(columns={"oof_proba": "home_win_proba(OOF)"}, inplace=True)

    st.dataframe(style_proba_table(hist, "home_win_proba(OOF)"), use_container_width=True)

    if df_next is not None:
        st.markdown("### Prochains matchs (aperçu)")
        nxt = df_next.sort_values("date").copy()
        if team_filter != "(toutes)":
            nxt = nxt[(nxt["home"] == team_filter) | (nxt["away"] == team_filter)]
        preview = nxt[["date", "home", "away", "home_win_proba"]].head(12).copy()
        st.dataframe(style_proba_table(preview, "home_win_proba"), use_container_width=True)

# ---------------------------
# Page: Next Matches
# ---------------------------
elif page == "📅 Next Matches":
    st.markdown("## Prochains matchs — prédictions")
    if df_next is None:
        st.warning("Aucune prédiction next trouvée. Lance `python scripts/predict_next.py`.")
        st.stop()

    nxt = df_next.sort_values("date").copy()
    if team_filter != "(toutes)":
        nxt = nxt[(nxt["home"] == team_filter) | (nxt["away"] == team_filter)]

    nxt["confidence_text"] = nxt["home_win_proba"].apply(lambda p: conf_label(float(p))[0])
    nxt["home_win_%"] = (nxt["home_win_proba"] * 100).round(1)

    # Pretty table
    st.markdown("### Tableau clair")
    table = nxt[["date", "home", "away", "home_win_%", "confidence_text"]].copy()

    def _pill(x):
        label, cls = conf_label({"Forte": 0.7, "Moyenne": 0.6, "Faible": 0.5}.get(x, 0.5))
        # use mapping of class based on text
        if x == "Forte":
            cls = "pill pill-green"
        elif x == "Moyenne":
            cls = "pill pill-yellow"
        else:
            cls = "pill pill-red"
        return f'<span class="{cls}">{x}</span>'

    # render as HTML (more polished)
    html = table.copy()
    html["confidence_text"] = html["confidence_text"].apply(_pill)
    html.rename(columns={"home_win_%": "Home Win (%)", "confidence_text": "Confiance"}, inplace=True)

    st.write(
        html.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    st.markdown("### Bar chart")
    chart = nxt.copy()
    chart["match"] = chart["home"] + " vs " + chart["away"]
    chart = chart.set_index("match")[["home_win_proba"]]
    st.bar_chart(chart)

    with st.expander("💡 Interprétation"):
        st.markdown(
            "- **Home Win (%)** : probabilité (selon le modèle) que l’équipe à domicile gagne.\n"
            "- **Confiance** : un repère visuel (seuils arbitraires) :\n"
            "  - 🟢 Forte ≥ 65%\n"
            "  - 🟡 Moyenne 55–65%\n"
            "  - 🔴 Faible < 55%\n\n"
            "Ce n’est pas une certitude : c’est une estimation basée sur l’historique + Elo + forme."
        )

# ---------------------------
# Page: Performance
# ---------------------------
elif page == "📈 Performance":
    st.markdown("## Performance du modèle (sur l’historique OOF)")
    if not has_oof:
        st.warning("Pas de `oof_proba` dans train_frame. Relance `python scripts/train_backtest.py`.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Accuracy (OOF)", fmt_pct(metrics["accuracy"]), "Seuil 0.5 sur oof_proba")
    with c2:
        kpi_card("LogLoss (OOF)", f"{metrics['logloss']:.3f}", "Plus bas = mieux")
    with c3:
        kpi_card("Brier (OOF)", f"{metrics['brier']:.3f}", "Plus bas = mieux")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Confusion matrix
    st.markdown("### Confusion matrix (OOF)")
    cm_df = pd.DataFrame(cm, index=["True: not homewin", "True: homewin"], columns=["Pred: not", "Pred: homewin"])
    st.dataframe(cm_df, use_container_width=True)

    # Calibration curve
    st.markdown("### Calibration (est-ce que 0.70 signifie vraiment ~70% ?)")
    y_true = df_train["y_homewin"].astype(int).values
    y_prob = df_train["oof_proba"].clip(1e-6, 1 - 1e-6).values
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    cal = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
    st.line_chart(cal.set_index("mean_predicted"))

    st.caption(
        "Courbe idéale : fraction_positive ≈ mean_predicted. "
        "Si la courbe est en dessous, le modèle est trop confiant ; au-dessus, pas assez confiant."
    )

    # Probability histogram
    st.markdown("### Distribution des probabilités (OOF)")
    hist = pd.DataFrame({"oof_proba": y_prob})
    st.bar_chart(hist["oof_proba"].value_counts(bins=20).sort_index())

    with st.expander("📌 À retenir"):
        st.markdown(
            "- **Accuracy** : performance “binaire” (home win vs pas home win)\n"
            "- **LogLoss / Brier** : qualité des probabilités (très important pour un modèle probabiliste)\n"
            "- **Calibration** : si le modèle dit 70%, est-ce que ça arrive ~70% du temps ?"
        )

# ---------------------------
# Page: Match Explainer
# ---------------------------
elif page == "🔎 Match Explainer":
    st.markdown("## Match Explainer")
    st.caption("Sélectionne un match : on affiche les features principales et on explique la proba.")

    mode = st.radio("Choisir la source", ["Historique (OOF)", "Prochains matchs"], horizontal=True)

    if mode == "Historique (OOF)":
        base = df_train.copy().sort_values("date", ascending=False)
        if team_filter != "(toutes)":
            base = base[(base["home"] == team_filter) | (base["away"] == team_filter)]

        base["label"] = base["date"].dt.strftime("%Y-%m-%d") + " — " + base["home"] + " vs " + base["away"]
        pick = st.selectbox("Match", base["label"].head(200).tolist())
        row = base[base["label"] == pick].iloc[0]

        p = float(row["oof_proba"]) if "oof_proba" in row else np.nan
        st.markdown(f"### {row['home']} vs {row['away']}")
        st.markdown(f"**Proba Home Win (OOF)** : `{p:.3f}` ({fmt_pct(p)})")

        # Show a compact feature table if importance exists
        if df_imp is not None:
            top_feats = df_imp.sort_values("abs_importance", ascending=False).head(8)["feature"].tolist()
        else:
            # fallback
            top_feats = [c for c in row.index if c.startswith("elo") or "pts" in c or "goal_diff" in c][:8]

        feat_rows = []
        for f in top_feats:
            if f in row.index:
                feat_rows.append((f, float(row[f]) if pd.notna(row[f]) else np.nan))
        feat_df = pd.DataFrame(feat_rows, columns=["feature", "value"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Explain using coefficient sign (logreg)
        if df_imp is not None:
            st.markdown("#### Explication (simple)")
            imp_map = df_imp.set_index("feature")["importance"].to_dict()
            bullets = []
            for f, v in feat_rows:
                coef = imp_map.get(f, None)
                if coef is None or pd.isna(v):
                    continue
                direction = "augmente" if coef > 0 else "diminue"
                bullets.append(f"- `{f}` = **{v:.3f}** → coefficient **{coef:.3f}** ⇒ {direction} la proba Home Win")
            if bullets:
                st.markdown("\n".join(bullets))
            else:
                st.info("Pas assez d’info pour expliquer (feature_importance manquant ou mismatch).")

    else:
        if df_next is None:
            st.warning("Aucune prédiction next trouvée. Lance `python scripts/predict_next.py`.")
            st.stop()

        base = df_next.copy().sort_values("date")
        if team_filter != "(toutes)":
            base = base[(base["home"] == team_filter) | (base["away"] == team_filter)]

        base["label"] = base["date"].dt.strftime("%Y-%m-%d") + " — " + base["home"] + " vs " + base["away"]
        pick = st.selectbox("Match", base["label"].tolist())
        row = base[base["label"] == pick].iloc[0]
        p = float(row["home_win_proba"])

        st.markdown(f"### {row['home']} vs {row['away']}")
        st.markdown(f"**Proba Home Win (next)** : `{p:.3f}` ({fmt_pct(p)})")

        label, cls = conf_label(p)
        st.markdown(f'<span class="{cls}">Confiance: {label}</span>', unsafe_allow_html=True)

        if df_imp is not None:
            top_feats = df_imp.sort_values("abs_importance", ascending=False).head(8)["feature"].tolist()
            feat_rows = []
            for f in top_feats:
                if f in row.index:
                    feat_rows.append((f, float(row[f]) if pd.notna(row[f]) else np.nan))
            feat_df = pd.DataFrame(feat_rows, columns=["feature", "value"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            st.markdown("#### Explication (simple)")
            imp_map = df_imp.set_index("feature")["importance"].to_dict()
            bullets = []
            for f, v in feat_rows:
                coef = imp_map.get(f, None)
                if coef is None or pd.isna(v):
                    continue
                direction = "augmente" if coef > 0 else "diminue"
                bullets.append(f"- `{f}` = **{v:.3f}** → coef **{coef:.3f}** ⇒ {direction} la proba")
            st.markdown("\n".join(bullets) if bullets else "—")

# ---------------------------
# Page: Model Insights
# ---------------------------
elif page == "🧠 Model Insights":
    st.markdown("## Model Insights")
    st.info(
        "Deux vues :\n"
        "- **Importance (coefficients)** = ce que le modèle pèse le plus (LogReg)\n"
        "- **Corrélation** = relation simple avec y_homewin (utile pour intuition, pas causalité)"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Feature importance")
        if df_imp is None:
            st.warning("feature_importance.csv absent. Relance `python scripts/train_backtest.py`.")
        else:
            top = df_imp.sort_values("abs_importance", ascending=False).head(15)
            st.dataframe(top[["feature", "importance", "abs_importance"]], use_container_width=True, hide_index=True)
            st.bar_chart(top.set_index("feature")["abs_importance"])

    with c2:
        st.markdown("### Correlation (safe)")
        if df_corr is None:
            st.warning("feature_correlation.csv absent. Lance `python scripts/analyze_features.py`.")
        else:
            topc = df_corr.sort_values("corr", ascending=False).head(15)
            st.dataframe(topc, use_container_width=True, hide_index=True)
            st.bar_chart(topc.set_index("feature")["corr"])

# ---------------------------
# Page: Glossary
# ---------------------------
else:
    st.markdown("## Glossary")
    st.markdown(
        "- **y_homewin** : 1 si l’équipe à domicile gagne, sinon 0\n"
        "- **OOF** : probas hors-échantillon (plus honnête que train direct)\n"
        "- **Elo** : rating de force d’équipe mis à jour match après match\n"
        "- **Form** : stats rolling sur les derniers matchs (points, buts…)\n"
        "- **LogLoss / Brier** : qualité probabiliste (plus bas = mieux)\n"
        "- **Calibration** : est-ce que 70% ≈ 70% dans la réalité ?\n"
    )