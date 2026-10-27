import os
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go, plotly.express as px
import core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = next((p for p in (os.path.join(HERE, "Nassau_Candy_Distributor.csv"), os.path.join(os.getcwd(), "Nassau_Candy_Distributor.csv"), os.path.join(HERE, "nassau_data.csv.gz")) if os.path.exists(p)), None)

st.set_page_config(page_title="Nassau Candy | Factory Optimizer", page_icon="🍬", layout="wide")
PINK, GOLD, TEAL, VIO = "#FF4F9A", "#FFC83D", "#2EE6C5", "#8B5CF6"
st.markdown(f"""<style>
.hero{{padding:22px 28px;border-radius:18px;background:linear-gradient(120deg,#3A1670,#C2185B 70%,#FF8A00);margin-bottom:14px}}
.hero h1{{margin:0;font-size:2rem}} .hero p{{margin:4px 0 0;opacity:.9}}
.card{{background:#1D1233;border:1px solid #2F2150;border-radius:14px;padding:14px 16px}}
.big{{font-size:1.9rem;font-weight:800}} .lbl{{opacity:.7;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase}}
.pill{{padding:2px 10px;border-radius:99px;font-size:.75rem;font-weight:700}}
.Low{{background:#12372F;color:{TEAL}}} .Medium{{background:#3B3112;color:{GOLD}}} .High{{background:#42142B;color:{PINK}}}
</style>""", unsafe_allow_html=True)

def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="lbl">{label}</div><div class="big">{value}</div><div style="opacity:.65;font-size:.8rem">{sub}</div></div>', unsafe_allow_html=True)

# ---------- data + models ----------
@st.cache_data(show_spinner="Loading data...")
def load(file):
    return core.load_clean(file)

@st.cache_resource(show_spinner="Training models...")
def fit(_df, key):
    return core.train_models(_df)

if DATA is None:
    st.error("Data file not found. Keep Nassau_Candy_Distributor.csv in the same folder as app.py."); st.stop()
df = load(DATA)
models, metrics, (Xte, yte) = fit(df, len(df))

with st.sidebar:
    st.markdown("## 🍬 Control Room")
    products = sorted(df['Product Name'].unique())
    product = st.selectbox("Product", products, index=products.index("Wonka Bar - Milk Chocolate") if "Wonka Bar - Milk Chocolate" in products else 0)
    region = st.selectbox("Destination region", ['All'] + sorted(df['Region'].unique()))
    mode = st.selectbox("Ship mode", ['All'] + sorted(df['Ship Mode'].unique()))
    w = st.slider("Optimization priority", 0, 100, 50, format="%d%% speed", help="0 = maximise profit, 100 = maximise speed") / 100
    st.caption(f"⚡ Speed {int(w*100)}%  |  💰 Profit {int((1-w)*100)}%")
    model_name = st.selectbox("Prediction model", list(models), index=2)
    topn = st.slider("Top-N recommendations", 1, 4, 3)
    freight = st.number_input("Freight cost ($ per unit per 1,000 km)", 0.0, 2.0, 0.05, 0.01,
        help="ASSUMPTION - the dataset has no freight cost. Profit impact = change in this modelled freight cost.")

sim = core.simulate(df, models[model_name], models['Random Forest'], freight, region=region, ship_mode=mode)
rec = core.rank(sim, w)
if rec.empty: st.warning("No orders match these filters."); st.stop()

st.markdown('<div class="hero"><h1>🍬 Nassau Candy · Factory Reallocation Optimizer</h1><p>Predict. Simulate. Reassign. Balance speed against profit before you move a single product.</p></div>', unsafe_allow_html=True)

K = core.kpis(rec)
c = st.columns(4)
with c[0]: card("Lead-time reduction", f"{K['Avg Lead Time Reduction (%)']}%", "avg of best move per product")
with c[1]: card("Profit impact stability", f"±{K['Profit Impact Stability (std of %)']}%", "std of profit change")
with c[2]: card("Scenario confidence", f"{K['Scenario Confidence Score']}/100", "model certainty + data support")
with c[3]: card("Recommendation coverage", f"{K['Recommendation Coverage (%)']}%", "products with a beneficial move")
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["🏭 Simulator", "🔀 What-If", "🏆 Recommendations", "🚨 Risk & Impact", "🔬 Data & Models"])

# ---------- 1 simulator ----------
with t1:
    p = rec[rec['Product Name'] == product].sort_values('pred_lead')
    cur = core.PRODUCT_FACTORY[product]
    st.subheader(f"{product}  ·  currently made at {cur}")
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(lat=[v[0] for v in core.FACTORIES.values()], lon=[v[1] for v in core.FACTORIES.values()],
            text=list(core.FACTORIES), mode='markers+text', textposition='top center',
            marker=dict(size=14, color=[PINK if f == cur else VIO for f in core.FACTORIES], line=dict(width=2, color='white'))))
        best = rec[(rec['Product Name'] == product) & (~rec['Is Current'])].iloc[0]
        la, lo = core.FACTORIES[cur]; lb, lob = core.FACTORIES[best.Factory]
        fig.add_trace(go.Scattergeo(lat=[la, lb], lon=[lo, lob], mode='lines', line=dict(width=3, color=GOLD, dash='dot')))
        fig.update_geos(scope='north america', bgcolor='rgba(0,0,0,0)', landcolor='#2A1B47', showcountries=True, countrycolor='#4B3A75', showsubunits=True, subunitcolor='#3A2A5E')
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Pink = current factory · gold line = best alternative at your priority ({best.Factory})")
    with b:
        fig = px.bar(p, x='pred_lead', y='Factory', orientation='h', color='Is Current', color_discrete_map={True: PINK, False: VIO},
                     labels={'pred_lead': 'Predicted lead time (days)'})
        fig.update_layout(height=430, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
        fig.update_xaxes(range=[max(0, p.pred_lead.min()-0.3), p.pred_lead.max()+0.3])
        st.plotly_chart(fig, use_container_width=True)
    show = p[['Factory', 'Is Current', 'avg_distance_km', 'pred_lead', 'lead_red_pct', 'profit_delta', 'confidence', 'risk']].round(2)
    show.columns = ['Factory', 'Current', 'Avg distance (km)', 'Pred. lead (d)', 'Lead Δ%', 'Freight saving ($)', 'Confidence', 'Risk']
    st.dataframe(show, hide_index=True, use_container_width=True)

# ---------- 2 what-if ----------
with t2:
    st.subheader("Current vs recommended assignment")
    alt = rec[(rec['Product Name'] == product) & (~rec['Is Current'])]
    choice = st.selectbox("Move to factory", ["★ Best recommendation"] + list(alt['Factory']))
    row = alt.iloc[0] if choice.startswith("★") else alt[alt.Factory == choice].iloc[0]
    now = rec[(rec['Product Name'] == product) & (rec['Is Current'])].iloc[0]
    m = st.columns(3)
    m[0].metric("Predicted lead time", f"{row.pred_lead:.2f} d", f"{row.pred_lead-now.pred_lead:+.3f} d", delta_color="inverse")
    m[1].metric("Avg shipping distance", f"{row.avg_distance_km:,.0f} km", f"{row.avg_distance_km-now.avg_distance_km:+,.0f} km", delta_color="inverse")
    m[2].metric("Freight cost", f"${row.freight:,.0f}", f"{row.freight-now.freight:+,.0f}", delta_color="inverse")
    cmp_ = pd.DataFrame({'Scenario': ['Current']*2 + [f'→ {row.Factory}']*2, 'Metric': ['Lead time (days)', 'Distance (×1000 km)']*2,
                         'Value': [now.pred_lead, now.avg_distance_km/1000, row.pred_lead, row.avg_distance_km/1000]})
    fig = px.bar(cmp_, x='Metric', y='Value', color='Scenario', barmode='group', color_discrete_sequence=[PINK, TEAL])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Moving **{product}** from **{cur}** to **{row.Factory}** would change average shipping distance by "
            f"**{-row.dist_red_pct:+.1f}%** and predicted lead time by **{-row.lead_red_pct:+.2f}%**, "
            f"with a modelled freight impact of **${row.profit_delta:+,.0f}** on the selected orders.")

# ---------- 3 recommendations ----------
with t3:
    st.subheader(f"Top-{topn} reassignment suggestions  ·  priority: ⚡{int(w*100)}% speed / 💰{int((1-w)*100)}% profit")
    best_per = rec[~rec['Is Current']].groupby('Product Name').head(topn)
    best_per = best_per[(best_per.lead_red_pct > 0) | (best_per.profit_delta > 0)]
    tbl = best_per.sort_values('score', ascending=False).head(25)[['Product Name', 'Factory', 'avg_distance_km', 'cur_dist', 'lead_red_pct', 'dist_red_pct', 'profit_delta', 'confidence', 'risk', 'score']].round(2)
    tbl.columns = ['Product', 'Move to', 'New dist (km)', 'Current dist (km)', 'Lead Δ%', 'Distance Δ%', 'Freight saving ($)', 'Confidence', 'Risk', 'Score']
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    fig = px.scatter(best_per, x='dist_red_pct', y='profit_delta', size='orders', color='risk', hover_name='Product Name',
                     hover_data=['Factory'], color_discrete_map={'Low': TEAL, 'Medium': GOLD, 'High': PINK},
                     labels={'dist_red_pct': 'Distance reduction (%)', 'profit_delta': 'Freight saving ($)'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=380)
    st.plotly_chart(fig, use_container_width=True)
    top1 = best_per.sort_values('score', ascending=False).iloc[0] if len(best_per) else None
    if top1 is not None:
        st.success(f"**Auto-summary:** the strongest move under your settings is **{top1['Product Name']} → {top1.Factory}** "
                   f"({top1.dist_red_pct:+.1f}% distance, ${top1.profit_delta:+,.0f} freight, {top1.confidence:.0f}/100 confidence, {top1.risk} risk). "
                   f"{K['Recommendation Coverage (%)']}% of products have at least one beneficial alternative.")

# ---------- 4 risk ----------
with t4:
    st.subheader("Profit-impact alerts & high-risk moves")
    mv = rec[~rec['Is Current']].copy()
    bad = mv[mv.profit_delta < 0].sort_values('profit_delta').head(10)
    risky = mv[mv.risk == 'High'].sort_values('confidence').head(10)
    a, b = st.columns(2)
    with a:
        st.markdown("**🔻 Moves that reduce profit**")
        for _, r in bad.iterrows():
            st.warning(f"{r['Product Name']} → {r.Factory}: freight +${-r.profit_delta:,.0f} ({r.profit_delta_pct:+.1f}% of gross profit)")
        if bad.empty: st.success("No profit-eroding moves in this selection.")
    with b:
        st.markdown("**⚠️ High-risk / low-confidence moves**")
        for _, r in risky.iterrows():
            st.error(f"{r['Product Name']} → {r.Factory}: confidence {r.confidence:.0f}/100, {r.extrapolation*100:.0f}% of orders outside historical distance range")
        if risky.empty: st.success("No high-risk moves in this selection.")
    fig = px.density_heatmap(mv, x='Factory', y='Product Name', z='profit_delta', histfunc='avg', color_continuous_scale='RdYlGn',
                             labels={'profit_delta': 'Freight saving ($)'})
    fig.update_layout(height=520, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ---------- 5 data & models ----------
with t5:
    st.subheader("Model comparison")
    st.dataframe(metrics, use_container_width=True)
    st.caption("RMSE penalises large errors · MAE = average error (days) · R² = variance explained.")
    d = df
    a, b = st.columns(2)
    with a:
        fig = px.box(d, x='Ship Mode', y='lead_days', color='Ship Mode', color_discrete_sequence=[PINK, GOLD, TEAL, VIO], labels={'lead_days': 'Lead time (days)'})
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig, use_container_width=True)
    with b:
        s = d.sample(min(3000, len(d)), random_state=1)
        fig = px.scatter(s, x='distance_km', y='lead_days', opacity=.35, color='Ship Mode', trendline=None, labels={'distance_km': 'Distance factory→customer (km)', 'lead_days': 'Lead time (days)'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig, use_container_width=True)
    corr = d[['lead_days', 'distance_km']].corr().iloc[0, 1]
    st.warning(f"**Data finding:** correlation between distance and lead time is only **{corr:.3f}**. Ship mode explains almost all lead-time variation, "
               "so predicted lead-time gains from moving factories are small and the recommendations are driven mainly by shorter distance and freight cost.")
    st.markdown("**Route clusters** (factory × destination state, KMeans)")
    rc = core.cluster_routes(d)
    fig = px.scatter(rc, x='dist', y='lead', size='orders', color=rc['cluster'].astype(str), hover_name='State/Province', hover_data=['factory'],
                     labels={'dist': 'Avg distance (km)', 'lead': 'Avg lead time (d)', 'color': 'Cluster'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig, use_container_width=True)
    with st.expander("Data preparation notes"):
        st.markdown("- Raw Ship−Order gaps are 900–1,640 days (three year-shifted blocks). Each block's minimum is treated as 0, leaving 0–11 real days.\n"
                    "- Customer coordinates are state/province centroids (dataset has none). Factory coordinates are from the project brief.\n"
                    "- Factory is derived from the product→factory table in the brief.\n"
                    "- Outliers beyond 3σ are removed before training. Models use distance-to-factory instead of a factory label so alternative factories can be scored.")
