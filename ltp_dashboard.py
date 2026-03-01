"""
ltp_dashboard.py — Le Top Paddock public picks tracker dashboard
"""
import os
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "ltp_picks.db")
BANKROLL_UNITS = 20  # 1u = 5% of bankroll → bankroll = 20u

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Le Top Paddock · Picks Tracker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
GOLD    = "#D4AF37"
GREEN   = "#27AE60"
RED     = "#E74C3C"
BG      = "#0e1117"
CARD_BG = "#161b25"
BORDER  = "#252b38"
TEXT    = "#FFFFFF"
MUTED   = "#8B8FA8"

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background-color: {BG}; }}
  [data-testid="stHeader"] {{ background-color: {BG}; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
  .metric-card {{
      background: {CARD_BG}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 20px 16px; text-align: center;
  }}
  .metric-value {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
  .metric-label {{ font-size: 0.75rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}
  .section-title {{
      font-size: 1.05rem; font-weight: 600; color: {TEXT};
      border-left: 3px solid {GOLD}; padding-left: 10px;
      margin: 32px 0 16px;
  }}
  .sport-header {{
      font-size: 1.1rem; font-weight: 700;
      margin-bottom: 18px; padding-bottom: 10px;
      border-bottom: 1px solid {BORDER};
  }}
  div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
  h1 {{ color: {TEXT} !important; }}
  h2, h3 {{ color: {TEXT} !important; }}
  .stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 500; }}
  .stTabs [aria-selected="true"] {{ color: {GOLD} !important; border-bottom-color: {GOLD} !important; }}
  .stMultiSelect [data-baseweb="tag"] {{ background-color: #1e2535; }}
</style>
""", unsafe_allow_html=True)

# ── Normalisation maps ─────────────────────────────────────────────────────────
BET_TYPE_MAP = {
    "parlay":        "Parlay",
    "combo":         "Parlay",
    "btts":          "Both Teams to Score",
    "total":         "Over / Under",
    "goals o/u":     "Goals Over / Under",
    "to draw":       "Draw",
    "to win":        "Match Winner",
    "moneyline":     "Match Winner",
    "handicap":      "Handicap",
    "shots":         "Player Prop",
    "player assist": "Player Prop",
    "player_prop":   "Player Prop",
    "other":         "Other",
}

LEAGUE_MAP = {
    # Soccer — European top flights
    "england": "Premier League",            "premier league": "Premier League",
    "spain": "La Liga",                     "la liga": "La Liga",
    "italy": "Serie A",                     "serie a": "Serie A",
    "germany": "Bundesliga",                "bundesliga": "Bundesliga",
    "france": "Ligue 1",                    "ligue 1": "Ligue 1",
    "otherenglish": "EFL Championship",     "efl": "EFL Championship",
    "efl championship": "EFL Championship", "championship": "EFL Championship",
    "turkey": "Süper Lig",                  "super lig": "Süper Lig",
    "netherlands": "Eredivisie",            "eredivisie": "Eredivisie",
    "portugal": "Primeira Liga",            "primeira liga": "Primeira Liga",
    "scotland": "Scottish Premiership",     "ireland": "League of Ireland",
    # Soccer — Americas / other
    "argentina": "Argentine Primera",
    "argentine primera división": "Argentine Primera",
    "denmark": "Danish Superliga",          "danish superliga": "Danish Superliga",
    # Soccer — European cups
    "champions": "Champions League",        "champions league": "Champions League",
    "ucl": "Champions League",
    "europa": "Europa League",              "europa league": "Europa League",
    "uel": "Europa League",
    "conference league": "Conference League", "uecl": "Conference League",
    # Soccer — multi-league parlays
    "soccer": "Multi-League Parlay",
    # Tennis — tiers (map everything to tier, not location)
    "atp 250": "ATP 250",                   "atp250": "ATP 250",
    "atp 500": "ATP 500",                   "atp500": "ATP 500",
    "atp 1000": "ATP 1000",                 "atp1000": "ATP 1000",
    "atp masters": "ATP 1000",
    "challenger": "ATP Challenger",         "atp challenger": "ATP Challenger",
    "tennis": "ATP 500",                    "atp tour": "ATP 500",
    "grand slam": "Grand Slam",
    "australian open": "Grand Slam",        "french open": "Grand Slam",
    "roland garros": "Grand Slam",          "wimbledon": "Grand Slam",
    "us open": "Grand Slam",
    # Specific ATP events → tier
    "acapulco open": "ATP 500",             "acapulco": "ATP 500",
    "rio open": "ATP 250",                  "buenos aires": "ATP 250",
    "argentina open": "ATP 250",
    "rotterdam open": "ATP 500",            "rotterdam": "ATP 500",
    "marseille": "ATP 250",                 "marseille open": "ATP 250",
    "dubai": "ATP 500",                     "dubai open": "ATP 500",
    "indian wells": "ATP 1000",             "miami open": "ATP 1000",
    "miami": "ATP 1000",                    "madrid open": "ATP 1000",
    "rome": "ATP 1000",                     "monte carlo": "ATP 1000",
    # Hockey
    "nhl": "NHL",
}

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "C$", "AUD": "A$"}
SPORT_COLOR      = {"Soccer": GOLD, "Tennis": "#4B9EFF", "Hockey": "#A8D8EA"}
SPORT_EMOJI      = {"Soccer": "⚽", "Tennis": "🎾", "Hockey": "🏒"}

SPORT_DEFAULTS = {
    "soccer": {"bankroll": 10000, "unit_pct": 0.05},
    "tennis": {"bankroll": 10000, "unit_pct": 0.05},
    "hockey": {"bankroll": 10000, "unit_pct": 0.05},
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_month(m: str) -> str:
    """Convert '2026-01' to 'January 2026'."""
    try:
        return datetime.strptime(m, "%Y-%m").strftime("%B %Y")
    except Exception:
        return m


def bucket_stake(u: float) -> str:
    if u <= 0.5:
        return "Small (≤0.5u)"
    elif u <= 1.25:
        return "Medium (0.75–1.25u)"
    else:
        return "Large (≥1.5u)"


# ── Data ───────────────────────────────────────────────────────────────────────
def _league_clean(row) -> str:
    league = row.get("league")
    sport  = row.get("sport", "")
    if not league or str(league).lower() in ("none", "null", ""):
        if sport == "tennis": return "ATP 500"
        if sport == "hockey": return "NHL"
        return "Unknown"
    mapped = LEAGUE_MAP.get(str(league).lower())
    return mapped if mapped else str(league).title()


@st.cache_data(ttl=60)
def load() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM picks ORDER BY date, id", conn, parse_dates=["date"])
    conn.close()
    # Round at source so every downstream sum/cumsum stays clean
    df["return_units"]   = df["return_units"].round(4)
    df["units"]          = df["units"].round(4)
    df["odds"]           = df["odds"].round(2)
    df["sport_label"]    = df["sport"].str.capitalize()
    df["bet_type_clean"] = df["bet_type"].map(
        lambda x: BET_TYPE_MAP.get(str(x).lower(), str(x).replace("_", " ").title() if x else "Other")
    )
    df["league_clean"] = df.apply(_league_clean, axis=1)
    df["dow"]          = df["date"].dt.day_name()
    df["month"]        = df["date"].dt.to_period("M").astype(str)
    df["stake_bucket"] = df["units"].map(bucket_stake)
    return df


@st.cache_data(ttl=60)
def load_monthly() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT substr(date,1,7) AS month, sport, SUM(return_units) AS units_won
        FROM picks WHERE result NOT IN ('pending')
        GROUP BY month, sport ORDER BY month
    """, conn)
    conn.close()
    return df


# ── UI helpers ─────────────────────────────────────────────────────────────────
def stat_card(col, value: str, label: str, color: str = TEXT):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def sport_stats(df: pd.DataFrame) -> dict:
    wins   = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    pushes = (df["result"] == "push").sum()
    staked = df["units"].sum()
    profit = df["return_units"].sum()
    roi    = profit / BANKROLL_UNITS * 100 if staked else 0
    wr     = wins / (wins + losses) * 100 if (wins + losses) else 0
    avg_o  = df[df["odds"].notna()]["odds"].mean() if df["odds"].notna().any() else 0
    return dict(total=len(df), wins=int(wins), losses=int(losses), pushes=int(pushes),
                win_rate=wr, units_staked=staked, units_profit=profit, roi=roi, avg_odds=avg_o)


def bar_chart(df_in, x_col, y_col, title, height=260):
    colors = [GREEN if v >= 0 else RED for v in df_in[y_col]]
    fig = go.Figure(go.Bar(
        x=df_in[x_col], y=df_in[y_col],
        marker_color=colors,
        hovertemplate="%{x}<br><b>%{y:+.2f}u</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=13)),
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=MUTED),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickformat="+.2f",
                   zeroline=True, zerolinecolor="#3a3f52", zerolinewidth=1),
        margin=dict(l=0, r=0, t=36, b=0),
        height=height,
    )
    return fig


def result_icon(r):
    return {"win": "✅ Win", "loss": "✖ Loss", "push": "➖ Push", "void": "🔵 Void", "pending": "⏳ Pending"}.get(r, r)


def calculate_compound(monthly_units, start_bankroll, unit_pct):
    bankroll = start_bankroll
    rows = []
    for month, units in monthly_units:
        begin     = bankroll
        unit_size = bankroll * unit_pct
        profit    = units * unit_size
        bankroll  = max(bankroll + profit, 0)
        rows.append({
            "Month":              month,
            "Beginning Bankroll": begin,
            "Units":              units,
            "Unit Size":          unit_size,
            "Profit":             profit,
            "Bankroll":           bankroll,
        })
    return rows


def render_compound_tab(sport_key, monthly_df, default_bankroll, default_unit_pct,
                        selected_sports=None, currency_sym="$"):
    c1, c2 = st.columns(2)
    start_br    = c1.number_input(
        f"Starting Bankroll ({currency_sym})", value=int(default_bankroll),
        step=1000, min_value=100, key=f"br_{sport_key or 'all'}",
    )
    unit_pct_in = c2.number_input(
        "Unit Size (%)", value=round(default_unit_pct * 100, 1),
        step=0.5, min_value=0.5, max_value=20.0, key=f"up_{sport_key or 'all'}",
    )
    unit_pct = unit_pct_in / 100

    if sport_key:
        mdf = monthly_df[monthly_df["sport"] == sport_key].copy()
    else:
        sports_to_include = selected_sports or ["soccer", "tennis", "hockey"]
        mdf = (monthly_df[monthly_df["sport"].isin(sports_to_include)]
               .groupby("month")["units_won"].sum().reset_index())
        mdf["sport"] = "combined"

    if mdf.empty:
        st.info("No settled picks yet.")
        return

    rows    = calculate_compound(list(zip(mdf["month"], mdf["units_won"])), float(start_br), unit_pct)
    comp_df = pd.DataFrame(rows)

    current_br   = comp_df["Bankroll"].iloc[-1]
    total_profit = current_br - start_br
    growth_pct   = total_profit / start_br * 100

    m1, m2, m3 = st.columns(3)
    stat_card(m1, f"{currency_sym}{current_br:,.0f}", "Current Bankroll",
              GREEN if current_br >= start_br else RED)
    profit_disp = (f"+{currency_sym}{total_profit:,.0f}" if total_profit >= 0
                   else f"-{currency_sym}{abs(total_profit):,.0f}")
    growth_disp = f"{'+' if growth_pct >= 0 else ''}{growth_pct:.1f}%"
    stat_card(m2, profit_disp, "Total Profit", GREEN if total_profit >= 0 else RED)
    stat_card(m3, growth_disp, "Growth",       GREEN if growth_pct >= 0 else RED)

    st.markdown("")

    # Month-by-month compound table
    disp = comp_df.copy()
    disp["Month"]              = disp["Month"].map(fmt_month)
    disp["Beginning Bankroll"] = disp["Beginning Bankroll"].map(lambda x: f"{currency_sym}{x:,.0f}")
    disp["Unit %"]             = f"{unit_pct * 100:.1f}%"
    disp["Unit Size"]          = disp["Unit Size"].map(lambda x: f"{currency_sym}{x:,.0f}")
    disp["Units Won"]          = disp["Units"].map(lambda x: f"{x:+.2f}u")
    disp["Profit"]             = disp["Profit"].map(
        lambda x: f"+{currency_sym}{x:,.0f}" if x >= 0 else f"-{currency_sym}{abs(x):,.0f}"
    )
    disp["Ending Bankroll"]    = disp["Bankroll"].map(lambda x: f"{currency_sym}{x:,.0f}")
    st.dataframe(
        disp[["Month", "Beginning Bankroll", "Unit %", "Unit Size", "Units Won", "Profit", "Ending Bankroll"]],
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
df_all  = load()
monthly = load_monthly()

# Header
st.markdown("# 🏆  Le Top Paddock")
st.markdown(
    f"<span style='color:{MUTED};font-size:0.9rem'>Verified picks record · Updated live</span>",
    unsafe_allow_html=True,
)

if df_all.empty:
    st.info("No picks data found.")
    st.stop()

df = df_all[df_all["result"] != "pending"].copy()

# Pending banner
pending_count = (df_all["result"] == "pending").sum()
if pending_count:
    st.markdown(
        f"<div style='background:#1a2030;border:1px solid {GOLD};border-radius:8px;"
        f"padding:10px 16px;margin-bottom:8px;font-size:0.9rem;color:{GOLD}'>"
        f"⏳  <b>{pending_count} pick(s) pending</b> — results update automatically</div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ── SECTION 1: Overall Record ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Overall Record</div>', unsafe_allow_html=True)

s = sport_stats(df)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
stat_card(c1, f"{s['wins']}W / {s['losses']}L", "Record")
stat_card(c2, f"{s['win_rate']:.1f}%",           "Win Rate",
          GREEN if s['win_rate'] >= 55 else (MUTED if s['win_rate'] >= 50 else RED))
stat_card(c3, f"{s['roi']:+.1f}%",              "ROI",         GREEN if s['roi'] > 0 else RED)
stat_card(c4, f"{s['units_profit']:+.2f}u",     "Net P&L",     GREEN if s['units_profit'] > 0 else RED)
stat_card(c5, f"{s['avg_odds']:.2f}",           "Avg Odds")
stat_card(c6, str(s['total']),                  "Total Bets")
stat_card(c7, f"{s['units_staked']:.1f}u",      "Units Staked")

st.markdown("")

# ── SECTION 2: Cumulative P&L ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Cumulative P&L</div>', unsafe_allow_html=True)

df_pl = df.sort_values(["sport", "date", "id"]).copy()
df_pl["cumul"]       = df_pl.groupby("sport")["return_units"].cumsum().round(2)
df_pl["sport_label"] = df_pl["sport"].str.capitalize()

fig_pl = px.line(
    df_pl, x="date", y="cumul", color="sport_label",
    color_discrete_map=SPORT_COLOR,
    template="plotly_dark",
)
fig_pl.update_traces(
    line=dict(width=2.5),
    hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>P&L: <b>%{y:+.2f}u</b><extra></extra>",
)
fig_pl.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
fig_pl.update_layout(
    plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, font=dict(color=MUTED),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Units", tickformat="+.2f"),
    legend=dict(title="", bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    margin=dict(l=0, r=0, t=12, b=0), height=320,
    hovermode="x unified",
)
st.plotly_chart(fig_pl, use_container_width=True)

# ── SECTION 3: Monthly Performance ────────────────────────────────────────────
st.markdown('<div class="section-title">Monthly Performance</div>', unsafe_allow_html=True)

monthly_summary = (
    df.groupby("month")
    .agg(
        bets=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        losses=("result", lambda x: (x == "loss").sum()),
        net=("return_units", "sum"),
        staked=("units", "sum"),
    )
    .reset_index()
)
monthly_summary["net"]      = monthly_summary["net"].round(4)
monthly_summary["win_rate"] = (
    monthly_summary["wins"] / (monthly_summary["wins"] + monthly_summary["losses"]) * 100
).round(1)
monthly_summary["roi"] = (monthly_summary["net"] / BANKROLL_UNITS * 100).round(1)
monthly_summary = monthly_summary.sort_values("month", ascending=False)

disp_m = monthly_summary[["month", "bets", "wins", "losses", "win_rate", "net", "roi"]].copy()
disp_m["month"]    = disp_m["month"].map(fmt_month)
disp_m.columns     = ["Month", "Bets", "W", "L", "Win %", "Net Units", "ROI %"]
disp_m["Win %"]    = disp_m["Win %"].map("{:.1f}%".format)
disp_m["Net Units"]= disp_m["Net Units"].map("{:+.2f}u".format)
disp_m["ROI %"]    = disp_m["ROI %"].map("{:+.1f}%".format)
st.dataframe(disp_m, use_container_width=True, hide_index=True)

# ── SECTION 4: By Sport ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">By Sport</div>', unsafe_allow_html=True)

# Sport summary cards
sport_cols = st.columns(3)
for col, sport in zip(sport_cols, ["soccer", "tennis", "hockey"]):
    sp = df[df["sport"] == sport]
    ss = sport_stats(sp)
    em = SPORT_EMOJI.get(sport.capitalize(), "")
    sc = SPORT_COLOR.get(sport.capitalize(), GOLD)
    pl_color = GREEN if ss["units_profit"] > 0 else RED
    sign = "+" if ss["units_profit"] >= 0 else ""
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.1rem;font-weight:700;color:{sc};margin-bottom:14px">{em}  {sport.capitalize()}</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<span style="color:{MUTED};font-size:0.82rem">Record</span>'
            f'<span style="font-weight:600">{ss["wins"]}W / {ss["losses"]}L</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<span style="color:{MUTED};font-size:0.82rem">Win Rate</span>'
            f'<span style="font-weight:600">{ss["win_rate"]:.1f}%</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<span style="color:{MUTED};font-size:0.82rem">ROI</span>'
            f'<span style="font-weight:600;color:{GREEN if ss["roi"]>0 else RED}">{ss["roi"]:+.1f}%</span></div>'
            f'<div style="display:flex;justify-content:space-between">'
            f'<span style="color:{MUTED};font-size:0.82rem">Net P&L</span>'
            f'<span style="font-weight:700;font-size:1.1rem;color:{pl_color}">{sign}{ss["units_profit"]:.2f}u</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# Sport detail tabs
tabs = st.tabs(["⚽  Soccer", "🎾  Tennis", "🏒  Hockey"])

for tab, sport in zip(tabs, ["soccer", "tennis", "hockey"]):
    with tab:
        sp = df[df["sport"] == sport].copy()
        if sp.empty:
            st.info(f"No {sport} picks recorded yet.")
            continue

        sp_color = SPORT_COLOR.get(sport.capitalize(), GOLD)

        # Cumulative P&L line
        sp_sorted = sp.sort_values(["date", "id"]).copy()
        sp_sorted["cumul"] = sp_sorted["return_units"].cumsum().round(2)

        fig_sp = go.Figure(go.Scatter(
            x=sp_sorted["date"], y=sp_sorted["cumul"],
            mode="lines+markers",
            line=dict(color=sp_color, width=2.5),
            marker=dict(
                color=[GREEN if r == "win" else (RED if r == "loss" else MUTED)
                       for r in sp_sorted["result"]],
                size=7,
                line=dict(color=CARD_BG, width=1),
            ),
            hovertemplate="%{x|%d %b %Y}<br>Cumulative P&L: <b>%{y:+.2f}u</b><extra></extra>",
            fill="tozeroy",
            fillcolor=f"rgba({int(sp_color[1:3],16)},{int(sp_color[3:5],16)},{int(sp_color[5:],16)},0.06)",
        ))
        fig_sp.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig_sp.update_layout(
            plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, font=dict(color=MUTED),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Cumulative Units", tickformat="+.2f"),
            margin=dict(l=0, r=0, t=12, b=0), height=280,
        )
        st.plotly_chart(fig_sp, use_container_width=True)

        # P&L by Bet Type + Day of Week
        col_a, col_b = st.columns(2)

        with col_a:
            bt = (sp.groupby("bet_type_clean")["return_units"]
                    .agg(["sum", "count"]).reset_index()
                    .rename(columns={"bet_type_clean": "Bet Type", "sum": "P&L", "count": "Bets"})
                    .sort_values("P&L", ascending=False))
            bt["P&L"] = bt["P&L"].round(2)
            st.plotly_chart(bar_chart(bt, "Bet Type", "P&L", "P&L by Bet Type"), use_container_width=True)

        with col_b:
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow = (sp.groupby("dow")["return_units"].sum()
                     .reindex(dow_order).fillna(0).reset_index()
                     .rename(columns={"dow": "Day", "return_units": "P&L"}))
            dow["P&L"] = dow["P&L"].round(2)
            st.plotly_chart(bar_chart(dow, "Day", "P&L", "P&L by Day of Week"), use_container_width=True)

        # P&L by Competition (soccer + tennis; hockey is all NHL so skip)
        if sport in ("soccer", "tennis"):
            lg = (sp.groupby("league_clean")["return_units"]
                    .agg(["sum", "count"]).reset_index()
                    .rename(columns={"league_clean": "Competition", "sum": "P&L", "count": "Bets"})
                    .sort_values("P&L", ascending=False))
            lg["P&L"] = lg["P&L"].round(2)
            st.plotly_chart(bar_chart(lg, "Competition", "P&L", "P&L by Competition", height=300),
                            use_container_width=True)

        # P&L by Stake Size (bucketed)
        bucket_order = ["Small (≤0.5u)", "Medium (0.75–1.25u)", "Large (≥1.5u)"]
        stake_grp = (sp.groupby("stake_bucket")["return_units"]
                       .agg(["sum", "count"]).reset_index()
                       .rename(columns={"stake_bucket": "Stake Size", "sum": "P&L", "count": "Bets"}))
        stake_grp["P&L"] = stake_grp["P&L"].round(2)
        stake_grp["Stake Size"] = pd.Categorical(
            stake_grp["Stake Size"], categories=bucket_order, ordered=True
        )
        stake_grp = stake_grp.sort_values("Stake Size")
        st.plotly_chart(bar_chart(stake_grp, "Stake Size", "P&L", "P&L by Stake Size"),
                        use_container_width=True)

# ── SECTION 5: All Bets History ───────────────────────────────────────────────
st.markdown('<div class="section-title">All Bets History</div>', unsafe_allow_html=True)

# Build month filter with formatted labels
month_vals        = sorted(df["month"].unique(), reverse=True)
month_label_map   = {m: fmt_month(m) for m in month_vals}
month_rev_map     = {v: k for k, v in month_label_map.items()}
month_display_opts = [month_label_map[m] for m in month_vals]

fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
sport_filter        = fc1.multiselect("Sport",    ["Soccer", "Tennis", "Hockey"],
                                       default=["Soccer", "Tennis", "Hockey"])
result_filter       = fc2.multiselect("Result",   ["Win", "Loss", "Push", "Void"],
                                       default=["Win", "Loss", "Push", "Void"])
bet_filter          = fc3.multiselect("Bet Type", sorted(df_all["bet_type_clean"].unique()))
month_filter_labels = fc4.multiselect("Month",    month_display_opts)
month_filter_raw    = [month_rev_map[l] for l in month_filter_labels]

df_table = df_all[df_all["sport_label"].isin(sport_filter)].copy()
df_table = df_table[df_table["result"].str.capitalize().isin(
    result_filter + ["Pending"] if "Pending" in result_filter else result_filter
)]
df_table = df_table[df_table["result"] != "pending"] if "Pending" not in result_filter else df_table
if bet_filter:
    df_table = df_table[df_table["bet_type_clean"].isin(bet_filter)]
if month_filter_raw:
    df_table = df_table[df_table["month"].isin(month_filter_raw)]

display = df_table.sort_values("date", ascending=False)[[
    "date", "sport_label", "league_clean", "description",
    "units", "bet_type_clean", "odds", "result", "return_units"
]].copy()

display["date"]         = display["date"].dt.strftime("%d %b %Y")
display["return_units"] = display.apply(
    lambda row: "—" if row["result"] == "pending" else f"{row['return_units']:+.2f}u", axis=1
)
display["odds"]   = display["odds"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
display["result"] = display["result"].map(result_icon)
display["units"]  = display["units"].map(lambda x: f"{round(x, 2):g}u")
display.columns   = ["Date", "Sport", "Competition", "Pick", "Stake", "Bet Type", "Odds", "Result", "P&L"]

if not df_table.empty:
    settled_t = df_table[df_table["result"].isin(["win", "loss", "push", "void"])]
    t_wins    = (settled_t["result"] == "win").sum()
    t_losses  = (settled_t["result"] == "loss").sum()
    t_net     = round(settled_t["return_units"].sum(), 2)
    st.markdown(
        f"<span style='color:{MUTED};font-size:0.85rem'>"
        f"Showing <b style='color:{TEXT}'>{len(settled_t)}</b> settled bets · "
        f"<b style='color:{GREEN if t_wins > t_losses else RED}'>{t_wins}W / {t_losses}L</b> · "
        f"Net: <b style='color:{GREEN if t_net >= 0 else RED}'>{t_net:+.2f}u</b></span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    height=480,
    column_config={
        "Date":        st.column_config.TextColumn("Date",        width=100),
        "Sport":       st.column_config.TextColumn("Sport",       width=80),
        "Competition": st.column_config.TextColumn("Competition", width=160),
        "Pick":        st.column_config.TextColumn("Pick",        width="large"),
        "Stake":       st.column_config.TextColumn("Stake",       width=70),
        "Bet Type":    st.column_config.TextColumn("Bet Type",    width=160),
        "Odds":        st.column_config.TextColumn("Odds",        width=65),
        "Result":      st.column_config.TextColumn("Result",      width=100),
        "P&L":         st.column_config.TextColumn("P&L",         width=80),
    },
)

# ── SECTION 6: Bankroll Compound Calculator ───────────────────────────────────
st.markdown('<div class="section-title">Bankroll Compound Calculator</div>', unsafe_allow_html=True)
st.markdown(
    f"<span style='color:{MUTED};font-size:0.85rem'>"
    f"Simulate how a bankroll compounds month-by-month following these picks. "
    f"Profits are reinvested at your chosen unit size each month.</span>",
    unsafe_allow_html=True,
)
st.markdown("")

currency     = st.selectbox("Currency", list(CURRENCY_SYMBOLS.keys()), key="calc_currency")
currency_sym = CURRENCY_SYMBOLS[currency]

# Combined is first tab (default open)
calc_tabs = st.tabs(["📊  Combined", "⚽  Soccer", "🎾  Tennis", "🏒  Hockey"])

with calc_tabs[0]:
    st.markdown(
        f"<span style='color:{MUTED};font-size:0.85rem'>"
        f"Pool units from all selected sports into one compounding bankroll.</span>",
        unsafe_allow_html=True,
    )
    sport_opts = st.multiselect(
        "Sports to include",
        options=["Soccer", "Tennis", "Hockey"],
        default=["Soccer", "Tennis", "Hockey"],
        key="combined_sports",
    )
    selected = [s.lower() for s in sport_opts]
    if selected:
        render_compound_tab(None, monthly, 10000, 0.05, selected_sports=selected,
                            currency_sym=currency_sym)
    else:
        st.info("Select at least one sport.")

for ctab, sport in zip(calc_tabs[1:], ["soccer", "tennis", "hockey"]):
    with ctab:
        d = SPORT_DEFAULTS[sport]
        render_compound_tab(sport, monthly, d["bankroll"], d["unit_pct"],
                            currency_sym=currency_sym)

# Footer
st.markdown("")
st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:0.75rem;"
    f"padding-top:16px;border-top:1px solid {BORDER}'>"
    f"Le Top Paddock · All records verified · Bet responsibly</div>",
    unsafe_allow_html=True,
)
