# 🍬 Nassau Candy – Factory Reallocation & Shipping Optimization

Interactive Streamlit dashboard that predicts shipping lead time, simulates moving products between factories, and recommends reassignments balancing speed and profit.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (no upload box; data loads automatically) |
| `core.py` | Data prep, models, simulation, ranking, clustering |
| `Nassau_Candy_Distributor.csv` | Dataset (must sit next to `app.py`) |
| `Nassau_Candy_Optimization.ipynb` | Google Colab notebook (EDA, models, simulation) |
| `.streamlit/config.toml` | Dark theme |
| `requirements.txt` | Dependencies |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push this repo to GitHub, go to share.streamlit.io, choose New app, select the repo, main file `app.py`.

## Key assumptions
- Raw Ship-Order gaps are year-shifted blocks; each block's minimum is treated as 0 days.
- Customer locations are state/province centroids; factory coordinates are from the project brief.
- Freight cost is not in the data; profit impact uses an adjustable $/unit/1000 km rate.
