"""Nassau Candy - Factory Reallocation & Shipping Optimization: core pipeline.
Shared by the Colab notebook and the Streamlit app."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack", "SweeTARTS": "Sugar Shack", "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack", "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory", "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory", "Hair Toffee": "The Other Factory", "Kazookles": "The Other Factory",
}
# Approximate geographic centroids (lat, lon) - customer location proxy (dataset has no coordinates)
STATE_XY = {
 'Alabama':(32.8,-86.8),'Alberta':(55.0,-115.0),'Arizona':(34.3,-111.7),'Arkansas':(34.9,-92.4),
 'British Columbia':(53.7,-127.6),'California':(37.2,-119.5),'Colorado':(39.0,-105.5),'Connecticut':(41.6,-72.7),
 'Delaware':(39.0,-75.5),'District of Columbia':(38.9,-77.0),'Florida':(28.6,-82.4),'Georgia':(32.7,-83.4),
 'Idaho':(44.4,-114.6),'Illinois':(40.0,-89.2),'Indiana':(39.9,-86.3),'Iowa':(42.1,-93.5),'Kansas':(38.5,-98.4),
 'Kentucky':(37.5,-85.3),'Louisiana':(31.1,-92.0),'Maine':(45.4,-69.2),'Manitoba':(55.0,-97.0),
 'Maryland':(39.0,-76.8),'Massachusetts':(42.3,-71.8),'Michigan':(44.3,-85.4),'Minnesota':(46.3,-94.3),
 'Mississippi':(32.7,-89.7),'Missouri':(38.4,-92.5),'Montana':(47.0,-109.6),'Nebraska':(41.5,-99.8),
 'Nevada':(39.3,-116.6),'New Brunswick':(46.5,-66.2),'New Hampshire':(43.7,-71.6),'New Jersey':(40.1,-74.7),
 'New Mexico':(34.4,-106.1),'New York':(42.9,-75.5),'Newfoundland and Labrador':(53.1,-57.7),
 'North Carolina':(35.5,-79.4),'North Dakota':(47.4,-100.5),'Nova Scotia':(45.0,-63.0),'Ohio':(40.3,-82.8),
 'Oklahoma':(35.6,-97.5),'Ontario':(50.0,-85.0),'Oregon':(43.9,-120.6),'Pennsylvania':(40.9,-77.8),
 'Prince Edward Island':(46.4,-63.4),'Quebec':(52.0,-71.5),'Rhode Island':(41.7,-71.5),'Saskatchewan':(54.0,-106.0),
 'South Carolina':(33.9,-80.9),'South Dakota':(44.4,-100.2),'Tennessee':(35.9,-86.4),'Texas':(31.5,-99.3),
 'Utah':(39.3,-111.7),'Vermont':(44.0,-72.7),'Virginia':(37.5,-78.8),'Washington':(47.4,-120.5),
 'West Virginia':(38.6,-80.6),'Wisconsin':(44.6,-89.9),'Wyoming':(43.0,-107.6),
}

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def normalise_lead(raw):
    """Raw (Ship-Order) gaps are 900-1640 days = three year-shifted blocks (~907/1273/1640).
    Each block's minimum is the 0-day baseline; remainder = true processing+transit days."""
    raw = np.asarray(raw)
    s = np.sort(np.unique(raw)); gaps = np.where(np.diff(s) > 100)[0]
    edges = [s[0]] + [s[i+1] for i in gaps]            # block starting points
    block = np.searchsorted(edges, raw, side='right') - 1
    return raw - np.array(edges)[block], block

def load_clean(path_or_df):
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
    df['raw_gap'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['lead_days'], df['date_block'] = normalise_lead(df['raw_gap'])
    df['Product Name'] = df['Product Name'].str.strip()
    df['factory'] = df['Product Name'].map(PRODUCT_FACTORY)
    df['cust_lat'] = df['State/Province'].map(lambda s: STATE_XY[s][0])
    df['cust_lon'] = df['State/Province'].map(lambda s: STATE_XY[s][1])
    df['f_lat'] = df['factory'].map(lambda f: FACTORIES[f][0]); df['f_lon'] = df['factory'].map(lambda f: FACTORIES[f][1])
    df['distance_km'] = haversine(df.cust_lat, df.cust_lon, df.f_lat, df.f_lon)
    df['unit_price'] = df['Sales']/df['Units']; df['margin_pct'] = df['Gross Profit']/df['Sales']*100
    return df

def remove_outliers(df, col='lead_days', k=3.0):
    z = (df[col]-df[col].mean())/df[col].std()
    return df[z.abs() <= k].copy()

CAT = ['Ship Mode', 'Region', 'Division', 'Product Name']
NUM = ['distance_km', 'Units', 'unit_price']

def make_prep():
    return ColumnTransformer([('num', StandardScaler(), NUM),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), CAT)])

def train_models(df, seed=42):
    """Models predict lead from product, destination, ship mode and *distance to origin factory*.
    Distance (not factory one-hot) is used so the model can score factory/product pairs never seen in history."""
    d = remove_outliers(df)
    X, y = d[CAT+NUM], d['lead_days']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    cands = {'Linear Regression': LinearRegression(),
             'Random Forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed),
             'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)}
    fitted, rows = {}, []
    for n, m in cands.items():
        p = Pipeline([('prep', make_prep()), ('m', m)]).fit(Xtr, ytr); fitted[n] = p
        pr = p.predict(Xte)
        rows.append({'Model': n, 'RMSE': np.sqrt(mean_squared_error(yte, pr)), 'MAE': mean_absolute_error(yte, pr), 'R2': r2_score(yte, pr)})
    res = pd.DataFrame(rows).set_index('Model').round(4)
    return fitted, res, (Xte, yte)

def ensemble_predict(pipe_rf, X):
    """Mean + std across Random-Forest trees -> uncertainty for confidence score."""
    Z = pipe_rf['prep'].transform(X)
    preds = np.stack([t.predict(Z) for t in pipe_rf['m'].estimators_])
    return preds.mean(0), preds.std(0)

def simulate(df, model_pipe, rf_pipe, freight_rate=0.05, product=None, region=None, ship_mode=None):
    """For every product x candidate factory: re-predict lead time for the product's historical orders
    as if shipped from that factory. freight_rate = $ per unit per 1000 km (ASSUMPTION; dataset has no freight cost)."""
    d = df.copy()
    if region and region != 'All': d = d[d['Region'] == region]
    if ship_mode and ship_mode != 'All': d = d[d['Ship Mode'] == ship_mode]
    if product: d = d[d['Product Name'] == product]
    out = []
    for prod, g in d.groupby('Product Name'):
        cur = PRODUCT_FACTORY[prod]
        hist = df[df['Product Name'] == prod]['distance_km']
        for fac, (la, lo) in FACTORIES.items():
            X = g[CAT+NUM].copy()
            X['distance_km'] = haversine(g.cust_lat, g.cust_lon, la, lo)
            lead = model_pipe.predict(X); _, sd = ensemble_predict(rf_pipe, X)
            out.append({'Product Name': prod, 'Division': g['Division'].iloc[0], 'Factory': fac, 'Is Current': fac == cur,
                        'orders': len(g), 'units': g['Units'].sum(), 'gross_profit': g['Gross Profit'].sum(),
                        'avg_distance_km': X['distance_km'].mean(), 'pred_lead': lead.mean(), 'pred_sd': sd.mean(),
                        'extrapolation': float(np.mean((X['distance_km'] < hist.min()) | (X['distance_km'] > hist.max()))),
                        'freight': (X['distance_km']/1000*g['Units'].values*freight_rate).sum()})
    s = pd.DataFrame(out)
    if s.empty: return s
    base = s[s['Is Current']].set_index('Product Name')
    s['cur_lead'] = s['Product Name'].map(base['pred_lead']); s['cur_freight'] = s['Product Name'].map(base['freight'])
    s['cur_dist'] = s['Product Name'].map(base['avg_distance_km'])
    s['lead_red_pct'] = (s.cur_lead - s.pred_lead)/s.cur_lead*100
    s['dist_red_pct'] = (s.cur_dist - s.avg_distance_km)/s.cur_dist*100
    s['profit_delta'] = s.cur_freight - s.freight                      # + = savings
    s['profit_delta_pct'] = s.profit_delta/s.gross_profit*100
    s['confidence'] = (100*(1 - s.pred_sd/max(float(s.pred_sd.max()), 1e-9)*0.5)*(1 - 0.5*s.extrapolation)).clip(0, 100)
    return s

def rank(s, speed_weight=0.5):
    """speed_weight 1 = pure speed (lead-time + distance), 0 = pure profit. Min-max within product."""
    s = s.copy()
    s['speed_score'] = 0.5*s.lead_red_pct + 0.5*s.dist_red_pct
    def mm(x): return (x-x.min())/(x.max()-x.min()) if x.max() > x.min() else x*0
    s['score'] = 0.0
    for p, i in s.groupby('Product Name').groups.items():
        s.loc[i, 'score'] = speed_weight*mm(s.loc[i, 'speed_score']) + (1-speed_weight)*mm(s.loc[i, 'profit_delta'])
    s['risk'] = np.where((s.profit_delta < 0) | (s.confidence < 50), 'High',
                 np.where((s.confidence < 70) | (s.extrapolation > .3), 'Medium', 'Low'))
    return s.sort_values(['Product Name', 'score'], ascending=[True, False])

def cluster_routes(df, k=4, seed=42):
    r = df.groupby(['factory', 'State/Province']).agg(orders=('Row ID', 'count'), lead=('lead_days', 'mean'),
        dist=('distance_km', 'mean'), profit=('Gross Profit', 'sum')).reset_index()
    r = r[r.orders >= 5].copy()
    Z = StandardScaler().fit_transform(r[['lead', 'dist', 'orders']])
    r['cluster'] = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    return r

def kpis(rec):
    top = rec[~rec['Is Current']].sort_values('score', ascending=False).groupby('Product Name').head(1)
    good = top[(top.lead_red_pct > 0) | (top.profit_delta > 0)]
    return {'Avg Lead Time Reduction (%)': round(top.lead_red_pct.mean(), 2),
            'Profit Impact Stability (std of %)': round(top.profit_delta_pct.std(ddof=0), 2),
            'Scenario Confidence Score': round(top.confidence.mean(), 1),
            'Recommendation Coverage (%)': round(100*len(good)/max(top['Product Name'].nunique(), 1), 1)}
