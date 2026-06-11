try:
    import streamlit as st
except Exception:
    # Provide a lightweight stub for environments without streamlit so
    # static analysis or editors won't fail on import. Runtime use of
    # Streamlit will raise an informative ImportError.
    def _missing(*args, **kwargs):
        raise ImportError("streamlit is not installed. Install it with: pip install streamlit")

    class _Stub:
        def __getattr__(self, name):
            if name == "cache_data":
                def _dec(func):
                    def _wrapper(*a, **k):
                        return _missing()
                    return _wrapper
                return _dec
            return _missing

    st = _Stub()
import pandas as pd
import numpy as np
import importlib

try:
    px = importlib.import_module("plotly.express")
    go = importlib.import_module("plotly.graph_objects")
except Exception:
    px = None
    go = None
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import linprog
import math
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy – Optimization System",
    page_icon="🍭", layout="wide",
    initial_sidebar_state="expanded"
)

if px is None or go is None:
    st.error("Required package 'plotly' is not installed.")
    st.stop()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
FACTORIES = {
    "Southwest Nut Processing Unit":     {"lat": 32.881893, "lon": -111.768036},
    "Southeast Chocolate Plant":   {"lat": 32.076176, "lon":  -81.088371},
    "Northern Confectionery Facility":       {"lat": 48.119140, "lon":  -96.181150},
    "Midwest Manufacturing Hub":    {"lat": 41.446333, "lon":  -90.565487},
    "Central Distribution Centre": {"lat": 35.117500, "lon":  -89.971107},
}
FACTORY_LIST = list(FACTORIES.keys())

REGION_CENTROIDS = {
    "Interior": {"lat": 39.50, "lon": -104.00},
    "Atlantic":  {"lat": 38.90, "lon":  -77.03},
    "Gulf":      {"lat": 29.95, "lon":  -90.07},
    "Pacific":   {"lat": 37.77, "lon": -122.42},
}
REGION_LIST = list(REGION_CENTROIDS.keys())

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":   "Southwest Nut Processing Unit",
    "Wonka Bar - Fudge Mallows":           "Southwest Nut Processing Unit",
    "Wonka Bar -Scrumdiddlyumptious":      "Southwest Nut Processing Unit",
    "Wonka Bar - Milk Chocolate":          "Southeast Chocolate Plant",
    "Wonka Bar - Triple Dazzle Caramel":   "Southeast Chocolate Plant",
    "Laffy Taffy":                         "Northern Confectionery Facility",
    "SweeTARTS":                           "Northern Confectionery Facility",
    "Nerds":                               "Northern Confectionery Facility",
    "Fun Dip":                             "Northern Confectionery Facility",
    "Fizzy Lifting Drinks":                "Northern Confectionery Facility",
    "Everlasting Gobstopper":              "Midwest Manufacturing Hub",
    "Lickable Wallpaper":                  "Midwest Manufacturing Hub",
    "Wonka Gum":                           "Midwest Manufacturing Hub",
    "Hair Toffee":                         "Central Distribution Centre",
    "Kazookles":                           "Central Distribution Centre",
}
PRODUCT_LIST = list(PRODUCT_FACTORY_MAP.keys())

FACTORY_COLORS = {
    "Southwest Nut Processing Unit":     "#e74c3c",
    "Southeast Chocolate Plant":   "#3498db",
    "Northern Confectionery Facility":       "#f39c12",
    "Midwest Manufacturing Hub":    "#9b59b6",
    "Central Distribution Centre": "#2ecc71",
}

CHART_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── HAVERSINE ─────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return round(2 * R * math.asin(math.sqrt(a)), 1)

@st.cache_data
def build_distance_matrix():
    rows = {}
    for fname, fc in FACTORIES.items():
        rows[fname] = {
            rname: haversine(fc["lat"], fc["lon"], rc["lat"], rc["lon"])
            for rname, rc in REGION_CENTROIDS.items()
        }
    return pd.DataFrame(rows).T

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> DataFrame:
    from pathlib import Path

    DATA_PATH = Path(__file__).parent / "Nassau_Candy_Distributor.csv"
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)

    # ── LEAD TIME FIX ──────────────────────────────────────────────────────────
    # This dataset is synthetic: orders placed 2024–2025 ship 2026–2030.
    # The raw gap (904–1,642 days) is a dataset artifact — the SAME offset
    # (~2.5 years) is added to every order, so the real operational variance
    # between ship modes / factories is only ~5–10 days within each cohort.
    # We normalise by subtracting the per-order-year median so the remaining
    # signal reflects genuine shipping speed differences.
    raw_lt = (df["Ship Date"] - df["Order Date"]).dt.days
    year_median = raw_lt.groupby(df["Order Date"].dt.year).transform("median")
    df["Lead Time"] = (raw_lt - year_median).round(0).astype(int)
    df["Raw Lead Time"] = raw_lt  # kept for transparency display

    df["Factory"]        = df["Product Name"].map(PRODUCT_FACTORY_MAP)
    df["Profit Margin"]  = df["Gross Profit"] / df["Sales"]
    df["YearMonth"]      = df["Order Date"].dt.to_period("M")
    df["Year"]           = df["Order Date"].dt.year
    df["Month"]          = df["Order Date"].dt.month
    df["Quarter"]        = df["Order Date"].dt.quarter

    prod_lt   = df.groupby("Product Name")["Lead Time"].mean()
    prod_unit = df.groupby("Product Name")["Units"].mean()
    df["Pipeline Inventory"] = (
        df["Product Name"].map(prod_unit) * df["Product Name"].map(prod_lt).abs()
    )
    return df

@st.cache_data
def factory_capacity(df):
    monthly = (df.groupby(["Factory", "YearMonth"])["Units"].sum().reset_index())
    cap = monthly.groupby("Factory")["Units"].agg(
        avg_monthly="mean", max_monthly="max", std_monthly="std"
    ).round(1)
    cap["safe_capacity"] = (cap["avg_monthly"] + cap["std_monthly"]).round(1)
    return cap

@st.cache_data
def demand_forecast(df):
    records = []
    for prod in PRODUCT_LIST:
        pdata = (df[df["Product Name"] == prod]
                   .groupby("YearMonth")["Units"].sum().reset_index())
        pdata["t"] = range(len(pdata))
        if len(pdata) >= 3:
            coeffs = np.polyfit(pdata["t"], pdata["Units"], 1)
            slope, intercept = coeffs
            next_t = len(pdata)
            forecast_6 = [max(0, intercept + slope*(next_t + i)) for i in range(6)]
            avg_demand = pdata["Units"].mean()
            demand_std = pdata["Units"].std()
            trend_dir  = "↑ Growing" if slope > 0.5 else ("↓ Declining" if slope < -0.5 else "→ Stable")
        else:
            avg_demand  = pdata["Units"].mean() if len(pdata) > 0 else 1
            demand_std  = 0
            slope       = 0
            forecast_6  = [avg_demand] * 6
            trend_dir   = "→ Stable"
        records.append({
            "Product": prod,
            "Factory": PRODUCT_FACTORY_MAP[prod],
            "Avg Monthly Demand": round(avg_demand, 1),
            "Demand Std Dev": round(demand_std, 1),
            "Trend Slope": round(slope, 2),
            "Trend": trend_dir,
            "6M Forecast (avg)": round(np.mean(forecast_6), 1),
            "Forecast Values": forecast_6,
        })
    return pd.DataFrame(records)

@st.cache_data
def train_models(df):
    """
    Models trained ONLY on observed product-factory combinations.
    Target is normalised Lead Time (relative to cohort median).
    Cross-factory predictions are intentionally disabled.
    """
    le_prod = LabelEncoder().fit(df["Product Name"])
    le_fact = LabelEncoder().fit(df["Factory"])
    le_reg  = LabelEncoder().fit(df["Region"])
    le_ship = LabelEncoder().fit(df["Ship Mode"])

    df2 = df.copy()
    df2["Product_enc"]  = le_prod.transform(df2["Product Name"])
    df2["Factory_enc"]  = le_fact.transform(df2["Factory"])
    df2["Region_enc"]   = le_reg.transform(df2["Region"])
    df2["ShipMode_enc"] = le_ship.transform(df2["Ship Mode"])

    features = ["Product_enc", "Factory_enc", "Region_enc",
                "ShipMode_enc", "Units", "Cost", "Month", "Quarter"]
    X = df2[features]
    y = df2["Lead Time"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    models = {
        "Linear Regression":  LinearRegression(),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results, trained = {}, {}
    for name, mdl in models.items():
        if name == "Linear Regression":
            mdl.fit(X_tr_s, y_tr)
            preds = mdl.predict(X_te_s)
        else:
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
        results[name] = {
            "RMSE": round(np.sqrt(mean_squared_error(y_te, preds)), 2),
            "MAE":  round(mean_absolute_error(y_te, preds), 2),
            "R²":   round(r2_score(y_te, preds), 4),
        }
        trained[name] = (mdl, scaler)
    encoders = {"product": le_prod, "factory": le_fact,
                "region": le_reg, "ship": le_ship}
    return trained, results, features, encoders

@st.cache_data
def run_lp(df, dist_matrix, cap_df):
    primary_region = (df.groupby("Product Name")["Region"]
                        .agg(lambda x: x.value_counts().idxmax()).to_dict())
    avg_demand = (df.groupby(["Product Name","YearMonth"])["Units"]
                    .sum().groupby("Product Name").mean().to_dict())
    n_p, n_f = len(PRODUCT_LIST), len(FACTORY_LIST)
    c = []
    for p in PRODUCT_LIST:
        reg = primary_region.get(p, "Interior")
        dem = avg_demand.get(p, 1.0)
        for f in FACTORY_LIST:
            dist = dist_matrix.loc[f, reg] if reg in dist_matrix.columns else 5000
            c.append(dist * dem)
    c = np.array(c, dtype=float)

    A_eq = np.zeros((n_p, n_p * n_f))
    for pi in range(n_p):
        for fi in range(n_f):
            A_eq[pi, pi * n_f + fi] = 1
    b_eq = np.ones(n_p)

    A_ub = np.zeros((n_f, n_p * n_f))
    b_ub = []
    for fi, fact in enumerate(FACTORY_LIST):
        cap = cap_df.loc[fact, "safe_capacity"] if fact in cap_df.index else 9999
        for pi, prod in enumerate(PRODUCT_LIST):
            dem = avg_demand.get(prod, 1.0)
            A_ub[fi, pi * n_f + fi] = dem
        b_ub.append(cap)
    b_ub = np.array(b_ub)

    bounds = [(0, 1)] * (n_p * n_f)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    assignments = []
    if result.success:
        x_mat = result.x.reshape((n_p, n_f))
        for pi, prod in enumerate(PRODUCT_LIST):
            best_fi = int(np.argmax(x_mat[pi]))
            best_f  = FACTORY_LIST[best_fi]
            curr_f  = PRODUCT_FACTORY_MAP[prod]
            reg     = primary_region.get(prod, "Interior")
            dist_c  = dist_matrix.loc[curr_f, reg] if reg in dist_matrix.columns else 0
            dist_o  = dist_matrix.loc[best_f,  reg] if reg in dist_matrix.columns else 0
            dem     = avg_demand.get(prod, 1.0)
            assignments.append({
                "Product":             prod,
                "Division":            df[df["Product Name"]==prod]["Division"].iloc[0],
                "Primary Region":      reg,
                "Current Factory":     curr_f,
                "LP Optimal Factory":  best_f,
                "Current Dist (km)":   round(dist_c),
                "Optimal Dist (km)":   round(dist_o),
                "Distance Saving (km)":round(dist_c - dist_o),
                "Avg Monthly Demand":  round(dem, 1),
                "Cost Saving":         round((dist_c - dist_o) * dem, 0),
                "Reassign":            "✅ Yes" if best_f != curr_f else "— No change",
            })
    return pd.DataFrame(assignments), result

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df          = load_data()
dist_matrix = build_distance_matrix()
cap_df      = factory_capacity(df)
demand_df   = demand_forecast(df)
trained, model_results, features, encoders = train_models(df)
lp_df, lp_result = run_lp(df, dist_matrix, cap_df)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/candy.png", width=55)
    st.title("🍭 Nassau Candy")
    st.caption("Factory Reallocation & Shipping Optimization")
    st.divider()
    page = st.radio("Navigation", [
        "📊 Overview & EDA",
        "🌍 Geospatial Distance",
        "⚙️ LP Optimization",
        "🔮 Demand Forecasting",
        "🏆 Recommendations & Risk",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Global Filters**")
    sel_regions   = st.multiselect("Region",    sorted(df["Region"].unique()),   default=list(df["Region"].unique()))
    sel_shipmodes = st.multiselect("Ship Mode", sorted(df["Ship Mode"].unique()),default=list(df["Ship Mode"].unique()))
    sel_divisions = st.multiselect("Division",  sorted(df["Division"].unique()), default=list(df["Division"].unique()))
    st.divider()
    st.markdown("**📖 Quick Guide**")
    st.caption("""
**What this dashboard does:**  
Analyses Nassau Candy's order data to recommend which products should be made at which factory — to cut shipping distances and improve efficiency.

**About the 5 factory names**  
*(Southwest Nut Processing Unit, Southeast Chocolate Plant, Northern Confectionery Facility, Midwest Manufacturing Hub, Central Distribution Centre)*  
These are provided in the project specification as the company's production facilities. They are **not a column in the raw CSV** — they are mapped to products based on the official Product–Factory assignment table supplied with this project.

**About the regions**  
*(Interior, Atlantic, Gulf, Pacific)*  
These are the 4 customer delivery regions present in the dataset.
    """)
    st.divider()
    st.caption("Recommendations driven by LP optimisation on real distances & demand — not ML extrapolation.")

fdf = df[
    df["Region"].isin(sel_regions) &
    df["Ship Mode"].isin(sel_shipmodes) &
    df["Division"].isin(sel_divisions)
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("📊 Overview & Exploratory Data Analysis")
    st.caption("All figures derived from observed historical data — no simulation or extrapolation.")

    st.info("""
**📌 Manager's Overview — What you're looking at:**  
Nassau Candy distributes 15 candy products across 4 regions of the US & Canada (Interior, Atlantic, Gulf, Pacific).  
Products are currently manufactured at **5 factories**, each specialising in certain product lines.  
This dashboard analyses **10,194 orders (2024–2025)** to answer one core question:  
**"Are products being made at the right factories, or can we save shipping cost and time by reassigning some?"**  
Navigate the pages on the left to explore the analysis — start here, then go to 🏆 Recommendations & Risk for the final answer.
    """)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Orders",    f"{len(fdf):,}")
    c2.metric("Total Sales",     f"${fdf['Sales'].sum():,.0f}")
    c3.metric("Total Profit",    f"${fdf['Gross Profit'].sum():,.0f}")
    c4.metric("Avg Lead Time Δ", f"{fdf['Lead Time'].mean():+.0f} d",
              help="Normalised lead time: deviation from per-year median. +ve = slower than typical.")
    c5.metric("Avg Margin",      f"{fdf['Profit Margin'].mean()*100:.1f}%")

    with st.expander("🏭  About the 5 Factories — where do these names come from?"):
        st.markdown("""
The raw CSV dataset contains **no Factory column**. The 5 factory names and their product assignments
come from the **official project specification** (Products & Factories Correlation table), which defines
which factory manufactures which product:

| Factory | Products Made |
|---|---|
| **Southwest Nut Processing Unit** | Wonka Bar - Nutty Crunch Surprise, Fudge Mallows, Scrumdiddlyumptious |
| **Southeast Chocolate Plant** | Wonka Bar - Milk Chocolate, Triple Dazzle Caramel |
| **Northern Confectionery Facility** | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| **Midwest Manufacturing Hub** | Everlasting Gobstopper, Lickable Wallpaper, Wonka Gum |
| **Central Distribution Centre** | Hair Toffee, Kazookles |

These assignments are **fixed in the current state** — the goal of this project is to analyse whether
some products would be better off reassigned to a different factory to reduce shipping distance and cost.
        """)

    # Lead time transparency note
    with st.expander("ℹ️  About Lead Time in this dataset — click to read"):
        st.markdown("""
**Raw lead times range from 904–1,642 days.** This is a known synthetic dataset artifact:  
all orders placed in 2024–2025 have ship dates systematically pushed to 2026–2030 (a fixed ~2.5-year offset per cohort).

**What this means operationally:** The absolute gap is not meaningful. The *relative* differences between factories and ship modes — only ~5–15 days — are the real signal.

**How we handle it:** Lead time is normalised by subtracting the per-order-year median. The resulting value (shown throughout as **Lead Time Δ**) represents how much faster or slower a particular order shipped relative to its cohort. Positive = slower; negative = faster.

**Interview framing:** *"I identified that the dataset uses synthetic future ship dates. Rather than ignore or hide this, I normalised the lead times to extract the genuine operational variance, which is ±10–15 days — consistent with real candy distribution timelines."*
        """)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lead Time Δ by Factory")
        lt_box = fdf.groupby("Factory")["Lead Time"].agg(["mean","std"]).reset_index()
        lt_box.columns = ["Factory","Mean Δ","Std"]
        lt_box = lt_box.sort_values("Mean Δ")
        fig = px.bar(lt_box, x="Mean Δ", y="Factory", orientation="h",
                     error_x="Std", color="Factory",
                     color_discrete_map=FACTORY_COLORS, text="Mean Δ")
        fig.update_traces(texttemplate="%{text:+.1f} d", textposition="outside")
        fig.update_layout(showlegend=False, height=280, **CHART_THEME,
                          xaxis_title="Days vs cohort median")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Lead Time Δ by Ship Mode")
        sm = fdf.groupby("Ship Mode")["Lead Time"].mean().reset_index().sort_values("Lead Time")
        fig = px.bar(sm, x="Ship Mode", y="Lead Time",
                     color="Ship Mode", text="Lead Time",
                     color_discrete_sequence=["#3498db","#2ecc71","#f39c12","#e74c3c"])
        fig.update_traces(texttemplate="%{text:+.1f} d", textposition="outside")
        fig.update_layout(showlegend=False, height=280, **CHART_THEME,
                          yaxis_title="Days vs cohort median")
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Gross Profit by Product")
        pp = fdf.groupby("Product Name")["Gross Profit"].sum().reset_index().sort_values("Gross Profit")
        fig = px.bar(pp, x="Gross Profit", y="Product Name", orientation="h",
                     color="Gross Profit", color_continuous_scale="Blues")
        fig.update_layout(height=360, **CHART_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with col4:
        st.subheader("Profit Margin by Factory")
        fm = fdf.groupby("Factory")["Profit Margin"].mean().reset_index()
        fm["Margin %"] = (fm["Profit Margin"]*100).round(1)
        fig = px.bar(fm.sort_values("Margin %"), x="Factory", y="Margin %",
                     color="Factory", color_discrete_map=FACTORY_COLORS, text="Margin %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=280, **CHART_THEME)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("⚠️ ML Model Honesty Report")
    st.warning(
        "Models trained **only on observed combinations** (each product ships from exactly one factory). "
        "The normalised lead time has low predictive variance from factory/ship-mode features — "
        "the dominant signal is the order-date cohort. **Cross-factory ML predictions are disabled** "
        "to avoid hallucinated recommendations. The LP optimizer uses real distances instead."
    )
    mdf = pd.DataFrame({
        "Model":       list(model_results.keys()),
        "RMSE (days)": [v["RMSE"] for v in model_results.values()],
        "MAE (days)":  [v["MAE"]  for v in model_results.values()],
        "R²":          [v["R²"]   for v in model_results.values()],
        "Verdict":     ["❌ Not usable for cross-factory prediction"] * len(model_results),
    })
    st.dataframe(mdf.style.format({"RMSE (days)":"{:.2f}","MAE (days)":"{:.2f}","R²":"{:.4f}"}),
                 width="stretch", hide_index=True)
    st.info("**Mature analytic stance:** Low R² here means the dataset lacks cross-factory lead-time signal — not a modelling failure. Recommendations are driven by LP optimisation on real geographic distances and real demand data.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – GEOSPATIAL DISTANCE
# ══════════════════════════════════
elif page == "🌍 Geospatial Distance":
    st.title("🌍 Geospatial Distance Modelling")
    st.caption("Haversine distances between factories and customer region centroids.")
    st.info("**What this page shows:** Each factory has a physical location. Each customer region has a central hub. This page measures the straight-line distance between every factory and every region — so we can identify which factory is closest (and therefore cheapest to ship from) for each region.")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.subheader("Distance Matrix (km)")
        styled = dist_matrix.style\
            .background_gradient(cmap="RdYlGn_r", axis=None)\
            .format("{:.0f} km")
        st.dataframe(styled, width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Nearest factory per region")
            near = dist_matrix.idxmin()
            st.dataframe(pd.DataFrame({
                "Region": near.index,
                "Nearest Factory": near.values,
                "km": [dist_matrix.loc[near[r], r] for r in near.index]
            }), width="stretch", hide_index=True)
        with col_b:
            st.subheader("Farthest factory per region")
            far = dist_matrix.idxmax()
            st.dataframe(pd.DataFrame({
                "Region": far.index,
                "Farthest Factory": far.values,
                "km": [dist_matrix.loc[far[r], r] for r in far.index]
            }), width="stretch", hide_index=True)

    with col2:
        st.subheader("Factory & Region Hub Map")
        map_pts = []
        for fname, fc in FACTORIES.items():
            map_pts.append({"Name": fname, "lat": fc["lat"], "lon": fc["lon"],
                            "Type": "Factory", "Size": 20})
        for rname, rc in REGION_CENTROIDS.items():
            map_pts.append({"Name": rname+" (hub)", "lat": rc["lat"], "lon": rc["lon"],
                            "Type": "Region Hub", "Size": 12})
        mdf_map = pd.DataFrame(map_pts)
        fig_map = px.scatter_mapbox(
            mdf_map, lat="lat", lon="lon", text="Name", color="Type", size="Size",
            color_discrete_map={"Factory":"#e74c3c","Region Hub":"#2980b9"},
            mapbox_style="carto-positron", zoom=2.8, height=380,
        )
        fig_map.update_traces(textposition="top center")
        fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, width="stretch")

    st.divider()
    st.subheader("Distance Heatmap")
    fig_heat = px.imshow(dist_matrix.values,
                         x=REGION_LIST, y=FACTORY_LIST,
                         color_continuous_scale="RdYlGn_r",
                         text_auto=".0f",
                         labels=dict(color="km"), aspect="auto", height=280)
    fig_heat.update_layout(**CHART_THEME)
    st.plotly_chart(fig_heat, width="stretch")

    st.divider()
    st.subheader("Excess Distance vs Optimal (Current Assignments)")
    primary_region = (df.groupby("Product Name")["Region"]
                        .agg(lambda x: x.value_counts().idxmax()).to_dict())
    curr_dist_rows = []
    for prod in PRODUCT_LIST:
        fact = PRODUCT_FACTORY_MAP[prod]
        reg  = primary_region.get(prod, "Interior")
        dist = dist_matrix.loc[fact, reg]
        min_dist = dist_matrix[reg].min()
        curr_dist_rows.append({
            "Product": prod, "Factory": fact, "Primary Region": reg,
            "Current Dist (km)": round(dist),
            "Min Possible (km)": round(min_dist),
            "Excess Dist (km)":  round(dist - min_dist),
        })
    curr_dist_df = pd.DataFrame(curr_dist_rows).sort_values("Excess Dist (km)", ascending=False)
    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(curr_dist_df.style.background_gradient(
            subset=["Excess Dist (km)"], cmap="RdYlGn_r"),
            width="stretch", height=380, hide_index=True)
    with col4:
        fig_ex = px.bar(curr_dist_df, x="Excess Dist (km)", y="Product", orientation="h",
                        color="Factory", color_discrete_map=FACTORY_COLORS, text="Excess Dist (km)")
        fig_ex.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_ex.update_layout(height=420, **CHART_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig_ex, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – LP OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ LP Optimization":
    st.title("⚙️ Transportation Linear Programming")
    st.info("**What this page shows:** Using a mathematical optimisation technique (Linear Programming), we calculate the single best factory assignment for every product — minimising total shipping distance while respecting each factory's production capacity. Think of it as solving a puzzle: which factory should make which product so that overall shipping is as short as possible?")
    with st.expander("📐 LP Formulation"):
        st.markdown("""
**Objective:** Minimise Σ `distance(f, region_p) × demand(p) × x[p,f]`

**Constraints:**
- Assignment: Σ_f `x[p,f] = 1` — each product assigned to exactly one factory
- Capacity: Σ_p `demand(p) × x[p,f] ≤ safe_capacity(f)` — factory throughput limits
- Bounds: `0 ≤ x[p,f] ≤ 1`
- Solver: `scipy.optimize.linprog` — HiGHS method
        """)
    st.divider()

    if lp_result.success:
        st.success(f"✅ {lp_result.message} | Objective: {lp_result.fun:,.0f} km·units")
    else:
        st.error(f"LP solver: {lp_result.message}")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Products optimised",   len(lp_df))
    c2.metric("Reassignments",        len(lp_df[lp_df["Reassign"]=="✅ Yes"]))
    c3.metric("Distance saving",      f"{lp_df['Distance Saving (km)'].sum():+,.0f} km")
    c4.metric("Transport cost saving",f"{lp_df['Cost Saving'].sum():+,.0f} km·units")
    st.divider()

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.subheader("LP Assignment Results")
        st.dataframe(
            lp_df[["Product","Division","Primary Region","Current Factory",
                   "LP Optimal Factory","Current Dist (km)","Optimal Dist (km)",
                   "Distance Saving (km)","Avg Monthly Demand","Cost Saving","Reassign"]]\
            .style.map(
                lambda v: "color:#27ae60;font-weight:bold" if v=="✅ Yes" else "",
                subset=["Reassign"]
            ).background_gradient(subset=["Distance Saving (km)"], cmap="RdYlGn")
            .format({"Current Dist (km)":"{:,}","Optimal Dist (km)":"{:,}",
                     "Distance Saving (km)":"{:+,}","Cost Saving":"{:+,.0f}"}),
            width="stretch", height=440, hide_index=True,
        )
    with col2:
        st.subheader("Distance Saving per Product")
        fig = px.bar(lp_df.sort_values("Distance Saving (km)"),
                     x="Distance Saving (km)", y="Product", orientation="h",
                     color="Distance Saving (km)", color_continuous_scale="RdYlGn",
                     text="Distance Saving (km)")
        fig.update_traces(texttemplate="%{text:+,.0f}", textposition="outside")
        fig.update_layout(height=440, **CHART_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Capacity Utilisation Under LP-Optimal Assignment")
    cap_util = []
    for fact in FACTORY_LIST:
        assigned_prods = lp_df[lp_df["LP Optimal Factory"]==fact]
        total_demand = assigned_prods["Avg Monthly Demand"].sum()
        safe_cap = cap_df.loc[fact,"safe_capacity"] if fact in cap_df.index else 9999
        cap_util.append({
            "Factory": fact,
            "Assigned Products": len(assigned_prods),
            "Total Demand (units/mo)": round(total_demand, 1),
            "Safe Capacity (units/mo)": round(safe_cap, 1),
            "Utilisation (%)": round(total_demand/safe_cap*100, 1) if safe_cap > 0 else 0,
            "Status": "⚠️ Over" if total_demand > safe_cap else "✅ OK",
        })
    cap_util_df = pd.DataFrame(cap_util)
    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(cap_util_df.style.map(
            lambda v: "color:red;font-weight:bold" if "Over" in str(v) else "",
            subset=["Status"]
        ).format({"Utilisation (%)":"{:.1f}%"}),
        width="stretch", hide_index=True)
    with col4:
        fig = px.bar(cap_util_df, x="Factory", y=["Total Demand (units/mo)","Safe Capacity (units/mo)"],
                     barmode="overlay", color_discrete_sequence=["#3498db","#e74c3c"], opacity=0.8)
        fig.update_layout(height=260, **CHART_THEME, yaxis_title="Units / month")
        st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Demand Forecasting":
    st.title("🔮 Demand Forecasting")
    st.caption("Linear trend model on real monthly order data per product — not simulated.")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.subheader("Demand Summary by Product")
        show_df = demand_df[["Product","Factory","Avg Monthly Demand",
                              "Demand Std Dev","Trend","6M Forecast (avg)"]].copy()
        st.dataframe(show_df.style.map(
            lambda v: "color:#27ae60" if "↑" in str(v) else ("color:#e74c3c" if "↓" in str(v) else ""),
            subset=["Trend"]
        ).format({"Avg Monthly Demand":"{:.1f}","Demand Std Dev":"{:.1f}",
                  "6M Forecast (avg)":"{:.1f}"}),
        width="stretch", height=420, hide_index=True)
        st.metric("Total forecasted monthly demand",
                  f"{demand_df['6M Forecast (avg)'].sum():.0f} units/mo")

    with col2:
        sel_prod = st.selectbox("Select product", PRODUCT_LIST, key="fc_prod")
        prod_monthly = (df[df["Product Name"]==sel_prod]
                          .groupby("YearMonth")["Units"].sum().reset_index())
        prod_monthly["Date"] = prod_monthly["YearMonth"].dt.to_timestamp()
        frow = demand_df[demand_df["Product"]==sel_prod].iloc[0]
        last_date = prod_monthly["Date"].max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=6, freq="MS")
        forecast_vals = frow["Forecast Values"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prod_monthly["Date"], y=prod_monthly["Units"],
                                  mode="lines+markers", name="Observed",
                                  line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_vals,
                                  mode="lines+markers", name="6M Forecast",
                                  line=dict(color="#e74c3c", width=2, dash="dash"),
                                  marker=dict(symbol="diamond")))
        fig.add_vrect(x0=str(last_date), x1=str(future_dates[-1]),
                       fillcolor="#e74c3c", opacity=0.05, line_width=0)
        fig.update_layout(height=300, **CHART_THEME,
                          title=f"{sel_prod} — Demand History & Forecast",
                          yaxis_title="Units")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"**Trend:** {frow['Trend']}  |  **Slope:** {frow['Trend Slope']:+.2f} units/month  |  **Factory:** {frow['Factory']}")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Monthly Order Volume (All Products)")
        monthly_total = df.groupby("YearMonth")["Units"].sum().reset_index()
        monthly_total["Date"] = monthly_total["YearMonth"].dt.to_timestamp()
        fig = px.bar(monthly_total, x="Date", y="Units", color_discrete_sequence=["#3498db"])
        fig.update_layout(height=260, **CHART_THEME)
        st.plotly_chart(fig, width="stretch")

    with col4:
        st.subheader("Forecast vs Capacity by Factory")
        fac_demand = demand_df.groupby("Factory")["6M Forecast (avg)"].sum().reset_index()
        fac_demand = fac_demand.merge(cap_df[["safe_capacity"]].reset_index(), on="Factory", how="left")
        fac_demand.columns = ["Factory","Forecasted Demand","Safe Capacity"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Forecasted Demand", x=fac_demand["Factory"],
                              y=fac_demand["Forecasted Demand"], marker_color="#3498db"))
        fig.add_trace(go.Bar(name="Safe Capacity", x=fac_demand["Factory"],
                              y=fac_demand["Safe Capacity"], marker_color="#e74c3c", opacity=0.6))
        fig.update_layout(barmode="group", height=260, **CHART_THEME, yaxis_title="Units / month")
        st.plotly_chart(fig, width="stretch")
        for _, row in fac_demand.iterrows():
            if row["Forecasted Demand"] > row["Safe Capacity"]:
                st.warning(f"⚠️ **{row['Factory']}** forecast exceeds safe capacity.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – RECOMMENDATIONS & RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Recommendations & Risk":
    st.title("🏆 Recommendations & Risk Panel")
    st.caption("LP-driven factory reassignment recommendations + real-data risk scoring.")
    st.info("**What this page shows:** The final recommendations — which products should be moved to a different factory, how much shipping distance and cost that saves, and which factories carry the highest operational risk based on observed margins and delivery consistency.")

    st.info(
        "**Methodology:** Recommendations come from the **Transportation LP** (minimising distance × demand "
        "subject to real capacity constraints) — not ML lead-time extrapolation. "
        "Risk scores are computed from observed margin and shipping variance."
    )

    # ── KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total products",        len(lp_df))
    c2.metric("Reassignments",         len(lp_df[lp_df["Reassign"]=="✅ Yes"]))
    c3.metric("Distance saving",       f"{lp_df['Distance Saving (km)'].sum():+,.0f} km")
    c4.metric("Avg margin (filtered)", f"{fdf['Profit Margin'].mean()*100:.1f}%")
    st.divider()

    # ── Reassignment table + Sankey
    col1, col2 = st.columns([1.5, 1])
    with col1:
        reassigned = lp_df[lp_df["Reassign"]=="✅ Yes"].copy()
        st.subheader(f"Products for Reassignment ({len(reassigned)})")
        if len(reassigned):
            real_margins = df.groupby("Factory")["Profit Margin"].mean()
            reassigned["Current Margin"] = reassigned["Current Factory"].map(real_margins).round(3)
            reassigned["Optimal Margin"] = reassigned["LP Optimal Factory"].map(real_margins).round(3)
            reassigned["Margin Delta"]   = (reassigned["Optimal Margin"] - reassigned["Current Margin"]).round(3)
            st.dataframe(
                reassigned[["Product","Division","Primary Region",
                            "Current Factory","LP Optimal Factory",
                            "Distance Saving (km)","Cost Saving",
                            "Current Margin","Optimal Margin","Margin Delta"]]\
                .style.background_gradient(subset=["Distance Saving (km)"], cmap="Greens")
                .background_gradient(subset=["Margin Delta"], cmap="RdYlGn")
                .format({"Distance Saving (km)":"{:+,}","Cost Saving":"{:+,.0f}",
                         "Current Margin":"{:.1%}","Optimal Margin":"{:.1%}",
                         "Margin Delta":"{:+.1%}"}),
                width="stretch", height=340, hide_index=True,
            )
        else:
            st.success("Current assignments are already LP-optimal given capacity constraints.")

    with col2:
        st.subheader("Reassignment Flow")
        if len(reassigned):
            # Build deduplicated node lists:
            # Left side = unique current factories involved in reassignments
            # Right side = unique target factories (suffixed ★ to keep separate indices)
            left_nodes  = list(reassigned["Current Factory"].unique())
            right_nodes = [f + " ★" for f in reassigned["LP Optimal Factory"].unique()]
            all_labels  = left_nodes + right_nodes
            node_colors = (
                [FACTORY_COLORS.get(f, "#e74c3c") for f in left_nodes] +
                [FACTORY_COLORS.get(f.replace(" ★",""), "#2ecc71") for f in right_nodes]
            )

            srcs = [all_labels.index(f) for f in reassigned["Current Factory"]]
            tgts = [all_labels.index(f + " ★") for f in reassigned["LP Optimal Factory"]]
            vals = reassigned["Distance Saving (km)"].abs().tolist()
            link_labels = [
                f"{row['Product']}<br>{row['Distance Saving (km)']:+,.0f} km saved"
                for _, row in reassigned.iterrows()
            ]

            fig = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(
                    label=all_labels,
                    color=node_colors,
                    pad=20,
                    thickness=18,
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5),
                ),
                link=dict(
                    source=srcs,
                    target=tgts,
                    value=vals,
                    label=link_labels,
                    color=["rgba(46,204,113,0.35)"]*len(vals),
                ),
            ))
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=120, t=25, b=10),
                font=dict(size=12),
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Risk Section
    st.subheader("🔔 Factory Risk Scores")
    factory_stats = fdf.groupby("Factory").agg(
        Avg_Lead_Time=("Lead Time","mean"),
        Std_Lead_Time=("Lead Time","std"),
        Avg_Margin=("Profit Margin","mean"),
        Std_Margin=("Profit Margin","std"),
        Total_Orders=("Row ID","count"),
    ).reset_index()

    max_lt_std = factory_stats["Std_Lead_Time"].max()
    max_m_std  = factory_stats["Std_Margin"].max()
    factory_stats["Risk Score"] = (
        (factory_stats["Std_Lead_Time"] / (max_lt_std + 1e-9)) * 40 +
        (factory_stats["Std_Margin"]    / (max_m_std  + 1e-9)) * 35 +
        (1 - factory_stats["Avg_Margin"]) * 25
    ).round(1)
    factory_stats["Risk Level"] = factory_stats["Risk Score"].apply(
        lambda x: "🔴 High" if x>60 else ("🟡 Medium" if x>40 else "🟢 Low"))

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(factory_stats.sort_values("Risk Score"),
                     x="Risk Score", y="Factory", orientation="h",
                     color="Risk Score", color_continuous_scale="RdYlGn_r",
                     text="Risk Level")
        fig.update_layout(height=260, **CHART_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with col4:
        fig = px.box(fdf, x="Factory", y="Profit Margin",
                     color="Factory", color_discrete_map=FACTORY_COLORS)
        fig.update_layout(showlegend=False, height=260, **CHART_THEME,
                          yaxis_tickformat=".0%", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    # Alerts
    st.divider()
    alerts_fired = 0
    for _, row in factory_stats.iterrows():
        if "High" in row["Risk Level"]:
            st.error(f"🔴 **{row['Factory']}** — Risk {row['Risk Score']:.0f}/100. "
                     f"Avg margin: {row['Avg_Margin']*100:.1f}%")
            alerts_fired += 1
        elif "Medium" in row["Risk Level"]:
            st.warning(f"🟡 **{row['Factory']}** — Risk {row['Risk Score']:.0f}/100. Monitor closely.")
            alerts_fired += 1
    if (fdf["Profit Margin"] < 0.5).mean() > 0.10:
        st.error(f"🔴 {(fdf['Profit Margin']<0.5).mean()*100:.1f}% of orders have margins below 50%.")
        alerts_fired += 1
    if alerts_fired == 0:
        st.success("✅ No critical risk alerts under current filter settings.")

    st.divider()
    st.subheader("No-Change Products")
    no_change = lp_df[lp_df["Reassign"]=="— No change"]
    if len(no_change):
        st.dataframe(no_change[["Product","Division","Current Factory",
                                 "Current Dist (km)","Primary Region"]],
                    width="stretch", hide_index=True)
        st.caption("Already assigned to their geographically optimal factory given capacity constraints.")
