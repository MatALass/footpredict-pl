Set-Location $PSScriptRoot\..
python -m pip install -e .
python scripts/build_dataset.py
python scripts/train_backtest.py
python scripts/analyze_features.py
python scripts/predict_next.py
streamlit run webapp/streamlit_app.py