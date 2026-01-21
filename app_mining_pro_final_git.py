import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Mining & Energy Risk Terminal", layout="wide")

COMMODITY_CONFIG = {
    "Gold": "GC=F", "Silver": "SI=F", "Carbon (KRBN)": "KRBN",
    "Copper": "HG=F", "Aluminum": "ALI=F", "Steel HRC": "HRC=F",
    "Brent Oil": "BZ=F", "WTI Oil": "CL=F", "Nat Gas (HH)": "NG=F",
    "Nat Gas (TTF)": "TTF=F",
}

COLOR_MAP = {
    "Gold": "#FFD700", "Silver": "#C0C0C0", "Copper": "#B87333",
    "Aluminum": "#D3D3D3", "Steel HRC": "#708090", "Carbon (KRBN)": "#2E8B57",
    "Brent Oil": "#1A1A1A", "WTI Oil": "#333333", "Nat Gas (HH)": "#4682B4",
    "Nat Gas (TTF)": "#5F9EA0"
}

# Added for the price window
COMMODITY_UNITS = {
    "Gold": "USD/t.oz", "Silver": "USD/t.oz", "Copper": "USD/lb",
    "Aluminum": "USD/mt", "Steel HRC": "USD/st", "Carbon (KRBN)": "USD/share",
    "Brent Oil": "USD/bbl", "WTI Oil": "USD/bbl", "Nat Gas (HH)": "USD/MMBtu",
    "Nat Gas (TTF)": "EUR/MWh"
}

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_regime_data():
    tickers = list(COMMODITY_CONFIG.values())
    df = yf.download(tickers, start="2022-01-01", auto_adjust=True)['Close']
    inv_map = {v: k for k, v in COMMODITY_CONFIG.items()}
    df = df.rename(columns=inv_map).ffill()
    return df[df.index >= "2023-01-01"]

df_prices = get_regime_data()
df_returns = 100 * np.log(df_prices / df_prices.shift(1))

def st_altair_line(df, title):
    # CRITICAL FIX: Only drop rows where EVERY asset is NaN.
    # This clips the leading empty year for the Sharpe chart 
    # but keeps the full 3-year history for the Horse Race.
    df_plot = df.dropna(how='all')
    
    if df_plot.empty:
        st.warning(f"No valid data points found for: {title}")
        return

    df_reset = df_plot.reset_index().melt('Date', var_name='Commodity', value_name='Value')
    num_years = df_plot.index.year.nunique()
    domain = list(COLOR_MAP.keys())
    range_colors = [COLOR_MAP.get(d, "#FFFFFF") for d in domain]
    
    # Altair will now automatically 'fit' the X-axis to the data range
    chart = alt.Chart(df_reset).mark_line().encode(
        x=alt.X('Date:T', 
                title=None,
                axis=alt.Axis(format='%Y', 
                              tickCount=num_years, 
                              labelAngle=0)),
        y=alt.Y('Value:Q', title=None, scale=alt.Scale(zero=False)),
        color=alt.Color('Commodity:N', scale=alt.Scale(domain=domain, range=range_colors))
    ).properties(title=title, height=450).interactive()
    
    st.altair_chart(chart, use_container_width=True)

# ==========================================
# 3. GLOBAL PERFORMANCE & EFFICIENCY
# ==========================================
st.title("The 'Horse Race' of major commodities (Base 100)") 
df_cum_rets = np.exp(df_returns.cumsum() / 100) * 100
st_altair_line(df_cum_rets, "Growth Attribution (Mathematical Price Path)")

st.title("Rolling Risk-Adjusted Efficiency (Sharpe)")
# The 252-day window is what caused the 'empty' gap on the left.
# st_altair_line will now clip that gap so the chart fills the screen.
roll_mean = df_returns.rolling(252).mean() * 252
roll_std = df_returns.rolling(252).std() * np.sqrt(252)
df_sharpe = (roll_mean / roll_std)
st_altair_line(df_sharpe, "252-Day Rolling Sharpe (Real-Time Relative Efficiency)")

# ==========================================
# 4. GLOBAL RISK LEADERBOARD
# ==========================================
st.divider()
st.title("Global Risk Leaderboard ($1M Position)")
if st.button("🚀 Run Global Risk Engine"):
    summary = []
    progress = st.progress(0)
    rng = np.random.default_rng(42) # Seed for reproducibility
    
    for i, name in enumerate(COMMODITY_CONFIG.keys()):
        try:
            rets = df_returns[name].dropna()
            # Correct Max Drawdown for Log Returns
            index_path = np.exp(rets.cumsum() / 100)
            max_dd = (index_path / index_path.cummax() - 1).min()
            
            model = arch_model(rets, mean='Constant', vol='Garch', p=1, q=1, dist='studentst')
            res = model.fit(disp='off')
            
            nu = max(float(res.params.get('nu', 10)), 2.1) # Safety floor
            sigma = float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]))
            
            shocks = stats.t.rvs(df=nu, size=10000, random_state=rng) / np.sqrt(nu / (nu - 2.0))
            sim_pnl = (sigma * shocks / 100.0) * 1000000
            
            v95, v99 = np.percentile(sim_pnl, 5), np.percentile(sim_pnl, 1)
            
            summary.append({
                "Commodity": name, "Max DD": max_dd,
                "95% VaR (Risk Threshold)": abs(v95),
                "99% ES (Avg Crash Depth)": abs(sim_pnl[sim_pnl <= v99].mean()),
                "Regime Sharpe": (rets.mean() * 252) / (rets.std() * np.sqrt(252))
            })
        except: continue
        progress.progress((i + 1) / len(COMMODITY_CONFIG))
    

    # Ensure keys in .format() match the dictionary keys exactly
    st.dataframe(
        pd.DataFrame(summary).sort_values("Regime Sharpe", ascending=False).style.format({
            "Max DD": "{:.2%}", 
            "95% VaR (Risk Threshold)": "${:,.0f}", 
            "99% ES (Avg Crash Depth)": "${:,.0f}", 
            "Regime Sharpe": "{:.2f}"
        }), 
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# 5. INDIVIDUAL EXHAUSTIVE DEEP-DIVE
# ==========================================
st.divider()
st.title("Individual Asset Exhaustive View")
target = st.selectbox("Select Asset", list(COMMODITY_CONFIG.keys()))
notional = st.number_input("Position Notional ($)", value=1000000, step=100000)

rng_ind = np.random.default_rng(42)
rets_ind = df_returns[target].dropna()

model_ind = arch_model(rets_ind, mean='Constant', vol='Garch', p=1, q=1, dist='studentst')
res_ind = model_ind.fit(disp='off')

nu_ind = max(float(res_ind.params.get('nu', 10)), 2.1)
sigma_ind = float(np.sqrt(res_ind.forecast(horizon=1).variance.iloc[-1, 0]))

shocks_ind = stats.t.rvs(df=nu_ind, size=10000, random_state=rng_ind) / np.sqrt(nu_ind / (nu_ind - 2.0))
sim_pnl_ind = (sigma_ind * shocks_ind / 100.0) * notional

v95, v99 = np.percentile(sim_pnl_ind, 5), np.percentile(sim_pnl_ind, 1)
es95 = sim_pnl_ind[sim_pnl_ind <= v95].mean()
es99 = sim_pnl_ind[sim_pnl_ind <= v99].mean()
index_path_ind = np.exp(rets_ind.cumsum() / 100)
max_dd_ind = (index_path_ind / index_path_ind.cummax() - 1).min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("95% Risk Threshold", f"-${abs(v95):,.0f}")
c2.metric("95% Avg Crash Depth", f"-${abs(es95):,.0f}")
c3.metric("99% Risk Threshold", f"-${abs(v99):,.0f}")
c4.metric("99% Avg Crash Depth", f"-${abs(es99):,.0f}")

st.metric("Regime Max Drawdown", f"{max_dd_ind:.2%}")

st.subheader("Quant Strategic Insights")
insight_data = {
    "Metric": ["Tail Sensitivity (ES/VaR)", "Fat Tail Check (DoF)", "Volatility Persistence"],
    "Value": [f"{(es99/v99):.2f}x", f"{nu_ind:.2f} DoF", "High" if (res_ind.params.get('alpha[1]', 0) + res_ind.params.get('beta[1]', 0) > 0.9) else "Low"]
}
st.table(pd.DataFrame(insight_data))

plt.style.use('dark_background')
fig_h, ax_h = plt.subplots(figsize=(10, 4))
fig_h.patch.set_facecolor('#0E1117') 
ax_h.set_facecolor('#0E1117')
sns.histplot(sim_pnl_ind, kde=True, color=COLOR_MAP.get(target, "#BF40BF"), alpha=0.7, ax=ax_h)
ax_h.axvline(v95, color='orange', ls='--', label='95% VaR')
ax_h.axvline(v99, color='cyan', ls='--', label='99% VaR')
ax_h.legend()
st.pyplot(fig_h)

# ==========================================
# 6. LIVE PRICE VERIFICATION WINDOW (NEW)
# ==========================================
st.divider()
st.title("Commodity Spot Price Tracker")
st.write("Current market values pulled live for model verification.")

latest_prices = df_prices.iloc[-1]
price_table = []
for commodity in COMMODITY_CONFIG.keys():
    price_table.append({
        "Commodity": commodity,
        "Unit": COMMODITY_UNITS.get(commodity, "N/A"),
        "Latest Price": latest_prices[commodity]
    })

st.table(pd.DataFrame(price_table).style.format({"Latest Price": "{:,.2f}"}))
