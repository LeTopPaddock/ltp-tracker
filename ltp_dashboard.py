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
  .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }}
  .metric-card {{
      background: {CARD_BG}; border: 1px solid {BORDER};
      border-radius: 12px; padding: 20px 16px; text-align: center;
  }}
  .metric-value {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
  .metric-label {{ font-size: 0.75rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}
  .section-title {{
      font-size: 1.05rem; font-weight: 600; color: {TEXT};
      border-left: 3px solid {GOLD}; padding-left: 10px; margin: 28px 0 14px;
  }}
  div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
  h1 {{ color: {TEXT} !important; }}
  h2, h3 {{ color: {TEXT} !important; }}
  .stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 500; }}
  .stTabs [aria-selected="true"] {{ color: {GOLD} !important; border-bottom-color: {GOLD} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Normalization maps ─────────────────────────────────────────────────────────
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

# Case-insensitive league map
LEAGUE_MAP = {
    # Soccer — domestic
    "england":           "Premier League",
    "premier league":    "Premier League",
    "spain":             "La Liga",
    "la liga":           "La Liga",
    "italy":             "Serie A",
    "serie a":           "Serie A",
    "germany":           "Bundesliga",
    "bundesliga":        "Bundesliga",
    "france":            "Ligue 1",
    "ligue 1":           "Ligue 1",
    "otherenglish":      "EFL",
    "efl":               "EFL",
    "efl championship":  "EFL",
    "argentina":         "Argentine Primera",
    "denmark":           "Danish Superliga",
    "soccer":            "Multi-League Parlay",
    # Soccer — European
    "champions":         "Champions League",
    "champions league":  "Champions League",
    "ucl":               "Champions League",
    "europa":            "Europa League",
    "europa league":     "Europa League",
    "uel":               "Europa League",
    "conference league": "Conference League",
    "uecl":              "Conference League",
    # Tennis
    "atp 500":           "ATP 500",
    "atp 250":           "ATP 250",
    "atp 1000":          "ATP Masters 1000",
    "challenger":        "ATP Challenger",
    "atp challenger":    "ATP Challenger",
    "tennis":            "ATP Tour",
    "acapulco open":     "Acapulco Open",
    "acapulco":          "Acapulco Open",
    "rio open":          "Rio Open",
    "rotterdam open":    "Rotterdam Open",
    "rotterdam":         "Rotterdam Open",
    "buenos aires":      "Argentina Open",
    "argentina open":    "Argentina Open",
    "marseille":         "Marseille Open",
    "marseille open":    "Marseille Open",
    "dubai":             "Dubai Open",
    "dubai open":        "Dubai Open",
    "indian wells":      "Indian Wells",
    "miami open":        "Miami Open",
    "miami":             "Miami Open",
    # Hockey
    "nhl":               "NHL",
}

SPORT_COLOR  = {"Soccer": GOLD, "Tennis": "#4B9EFF", "Hockey": "#A8D8EA"}
SPORT_EMOJI  = {"Soccer": "⚽", "Tennis": "🎾", "Hockey": "🏒"}

# Default bankroll settings per sport
SPORT_DEFAULTS = {
    "soccer":  {"bankroll": 20000, "unit_pct": 0.05},
    "tennis":  {"bankroll": 10000, "unit_pct": 0.01},
    "hockey":  {"bankroll": 20000, "unit_pct": 0.05},
}


# ── Data ───────────────────────────────────────────────────────────────────────
def _league_clean(row) -> str:
    league = row.get("league")
    sport  = row.get("sport", "")
    if not league or str(league).lower() in ("none", "null", ""):
        if sport == "tennis": return "ATP Tour"
        if sport == "hockey": return "NHL"
        return "Unknown"
    mapped = LEAGUE_MAP.get(str(league).lower())
    if mapped:
        return mapped
    return str(league).title()


@st.cache_data(ttl=60)
def load() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM picks ORDER BY date, id", conn, parse_dates=["date"])
    conn.close()
    df["sport_label"]    = df["sport"].str.capitalize()
    df["bet_type_clean"] = df["bet_type"].map(
        lambda x: BET_TYPE_MAP.get(str(x).lower(), str(x).replace("_"," ").title() if x else "Other")
    )
    df["league_clean"] = df.apply(_league_clean, axis=1)
    df["dow"]          = df["date"].dt.day_name()
    df["month"]        = df["date"].dt.to_period("M").astype(str)
    return df


@st.cache_data(ttl=60)
def load_monthly() -> pd.DataFrame:
    """Monthly units won per sport for compound calculator."""
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


# ── Chart helpers ──────────────────────────────────────────────────────────────
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
    roi    = profit / staked * 100 if staked else 0
    wr     = wins / (wins + losses) * 100 if (wins + losses) else 0
    avg_o  = df[df["odds"].notna()]["odds"].mean() if df["odds"].notna().any() else 0
    return dict(total=len(df), wins=int(wins), losses=int(losses), pushes=int(pushes),
                win_rate=wr, units_staked=staked, units_profit=profit, roi=roi, avg_odds=avg_o)


def bar_chart(df_in, x_col, y_col, title):
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
        height=280,
    )
    return fig


def result_icon(r):
    return {"win": "✅ Win", "loss": "✖ Loss", "push": "➖ Push", "pending": "⏳ Pending"}.get(r, r)


# ── Compound calculator ────────────────────────────────────────────────────────
def calculate_compound(monthly_units: list, start_bankroll: float, unit_pct: float) -> list:
    """Calculate month-by-month compound bankroll growth."""
    bankroll = start_bankroll
    rows = []
    for month, units in monthly_units:
        unit_size = bankroll * unit_pct
        profit    = units * unit_size
        bankroll  = max(bankroll + profit, 0)
        rows.append({
            "Month":     month,
            "Units":     units,
            "Unit Size": unit_size,
            "Profit":    profit,
            "Bankroll":  bankroll,
        })
    return rows


def render_compound_tab(sport_key: str | None, monthly_df: pd.DataFrame,
                        default_bankroll: float, default_unit_pct: float,
                        selected_sports: list | None = None):
    """Render a compound calculator tab for one sport or a combined selection."""
    c1, c2 = st.columns([1, 1])
    start_br   = c1.number_input("Starting Bankroll ($)", value=int(default_bankroll),
                                  step=1000, min_value=100, key=f"br_{sport_key or 'all'}")
    unit_pct_in = c2.number_input("Unit Size (%)", value=round(default_unit_pct * 100, 1),
                                   step=0.5, min_value=0.5, max_value=20.0,
                                   key=f"up_{sport_key or 'all'}")
    unit_pct = unit_pct_in / 100

    # Filter monthly data
    if sport_key:
        mdf = monthly_df[monthly_df["sport"] == sport_key].copy()
    else:
        sports_to_include = selected_sports or ["soccer", "tennis", "hockey"]
        mdf = (monthly_df[monthly_df["sport"].isin(sports_to_include)]
               .groupby("month")["units_won"].sum().reset_index()
               .rename(columns={"units_won": "units_won"}))
        mdf["sport"] = "combined"

    if mdf.empty:
        st.info("No settled picks yet to calculate compounding.")
        return

    monthly_list = list(zip(mdf["month"], mdf["units_won"]))
    rows = calculate_compound(monthly_list, float(start_br), unit_pct)
    comp_df = pd.DataFrame(rows)

    # Summary metrics
    current_br  = comp_df["Bankroll"].iloc[-1]
    total_profit = current_br - start_br
    growth_pct   = total_profit / start_br * 100
    p_sign = "+" if total_profit >= 0 else ""

    m1, m2, m3 = st.columns(3)
    stat_card(m1, f"${current_br:,.0f}", "Current Bankroll", GREEN if current_br >= start_br else RED)
    stat_card(m2, f"{p_sign}${total_profit:,.0f}", "Total Profit", GREEN if total_profit >= 0 else RED)
    stat_card(m3, f"{p_sign}{growth_pct:.1f}%", "Growth", GREEN if growth_pct >= 0 else RED)

    st.markdown("")

    # Bankroll growth chart
    fig = go.Figure(go.Scatter(
        x=comp_df["Month"], y=comp_df["Bankroll"],
        mode="lines+markers",
        line=dict(color=GOLD, width=2.5),
        marker=dict(color=GOLD, size=7),
        hovertemplate="<b>%{x}</b><br>Bankroll: $%{y:,.0f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(212,175,55,0.08)",
    ))
    fig.add_hline(y=start_br, line_dash="dot", line_color=BORDER, line_width=1)
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=MUTED),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Bankroll ($)", tickformat="$,.0f"),
        margin=dict(l=0, r=0, t=12, b=0), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Month-by-month table
    with st.expander("Month-by-month breakdown"):
        disp = comp_df.copy()
        disp["Units"]     = disp["Units"].map(lambda x: f"{x:+.2f}u")
        disp["Unit Size"] = disp["Unit Size"].map(lambda x: f"${x:,.0f}")
        disp["Profit"]    = disp["Profit"].map(lambda x: f"{'+' if x >= 0 else ''}${x:,.0f}")
        disp["Bankroll"]  = disp["Bankroll"].map(lambda x: f"${x:,.0f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ── App ────────────────────────────────────────────────────────────────────────
df_all   = load()
monthly  = load_monthly()

st.markdown("# 🏆  Le Top Paddock")
st.markdown(f"<span style='color:{MUTED};font-size:0.9rem'>Verified picks record · Updated live</span>",
            unsafe_allow_html=True)

if df_all.empty:
    st.info("No picks data found.")
    st.stop()

df = df_all[df_all["result"] != "pending"].copy()

pending_count = (df_all["result"] == "pending").sum()
if pending_count:
    st.markdown(
        f"<div style='background:#1a2030;border:1px solid {GOLD};border-radius:8px;"
        f"padding:10px 16px;margin-bottom:16px;font-size:0.9rem;color:{GOLD}'>"
        f"⏳  <b>{pending_count} pick(s) pending</b> — results will update automatically</div>",
        unsafe_allow_html=True,
    )

s = sport_stats(df)

# ── Overall stats ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Overall Record — All Sports</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
stat_card(c1, f"{s['wins']}W / {s['losses']}L",   "Record")
stat_card(c2, f"{s['win_rate']:.1f}%",             "Win Rate",
          GREEN if s['win_rate'] >= 55 else (MUTED if s['win_rate'] >= 50 else RED))
stat_card(c3, f"{s['roi']:+.1f}%",                 "ROI",
          GREEN if s['roi'] > 0 else RED)
stat_card(c4, f"{s['units_profit']:+.2f}u",        "Total P&L",
          GREEN if s['units_profit'] > 0 else RED)
stat_card(c5, f"{s['avg_odds']:.2f}",              "Avg Odds")
stat_card(c6, str(s['total']),                     "Total Bets")
stat_card(c7, f"{s['units_staked']:.2f}u",         "Units Staked")

st.markdown("")

# ── Cumulative P&L chart ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Cumulative P&L by Sport</div>', unsafe_allow_html=True)

df_pl = df.sort_values(["sport", "date", "id"]).copy()
df_pl["cumul"] = df_pl.groupby("sport")["return_units"].cumsum()
df_pl["sport_label"] = df_pl["sport"].str.capitalize()

fig_pl = px.line(
    df_pl, x="date", y="cumul", color="sport_label",
    color_discrete_map=SPORT_COLOR,
    template="plotly_dark",
)
fig_pl.update_traces(line=dict(width=2.5),
                     hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>P&L: <b>%{y:+.2f}u</b><extra></extra>")
fig_pl.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
fig_pl.update_layout(
    plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
    font=dict(color=MUTED),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Units", tickformat="+.2f"),
    legend=dict(title="", bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    margin=dict(l=0, r=0, t=12, b=0), height=340,
    hovermode="x unified",
)
st.plotly_chart(fig_pl, use_container_width=True)

# ── Sport summary cards ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">By Sport</div>', unsafe_allow_html=True)
cols = st.columns(3)

for col, sport in zip(cols, ["soccer", "tennis", "hockey"]):
    sp = df[df["sport"] == sport]
    ss = sport_stats(sp)
    em = SPORT_EMOJI[sport.capitalize()]
    sc = SPORT_COLOR[sport.capitalize()]
    pl_color = GREEN if ss["units_profit"] > 0 else RED
    sign = "+" if ss["units_profit"] >= 0 else ""
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.25rem;font-weight:700;color:{sc};margin-bottom:14px">{em}  {sport.capitalize()}</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<span style="color:{MUTED};font-size:0.8rem">Record</span>'
            f'<span style="font-weight:600">{ss["wins"]}W / {ss["losses"]}L</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<span style="color:{MUTED};font-size:0.8rem">Win Rate</span>'
            f'<span style="font-weight:600">{ss["win_rate"]:.1f}%</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<span style="color:{MUTED};font-size:0.8rem">ROI</span>'
            f'<span style="font-weight:600;color:{GREEN if ss["roi"]>0 else RED}">{ss["roi"]:+.1f}%</span></div>'
            f'<div style="display:flex;justify-content:space-between">'
            f'<span style="color:{MUTED};font-size:0.8rem">P&L</span>'
            f'<span style="font-weight:700;font-size:1.1rem;color:{pl_color}">{sign}{ss["units_profit"]:.2f}u</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Per-sport detail tabs ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Detailed Stats by Sport</div>', unsafe_allow_html=True)
tabs = st.tabs(["⚽  Soccer", "🎾  Tennis", "🏒  Hockey"])

for tab, sport in zip(tabs, ["soccer", "tennis", "hockey"]):
    with tab:
        sp = df[df["sport"] == sport].copy()
        if sp.empty:
            st.info(f"No {sport} picks recorded yet.")
            continue

        ss = sport_stats(sp)
        r1, r2, r3, r4, r5 = st.columns(5)
        stat_card(r1, f"{ss['wins']}W / {ss['losses']}L", "Record")
        stat_card(r2, f"{ss['win_rate']:.1f}%", "Win Rate",
                  GREEN if ss["win_rate"] >= 55 else (MUTED if ss["win_rate"] >= 50 else RED))
        stat_card(r3, f"{ss['roi']:+.1f}%", "ROI", GREEN if ss["roi"] > 0 else RED)
        stat_card(r4, f"{ss['units_profit']:+.2f}u", "P&L", GREEN if ss["units_profit"] > 0 else RED)
        stat_card(r5, f"{ss['avg_odds']:.2f}", "Avg Odds")

        st.markdown("")

        # Cumulative P&L line
        sp_cumul = sp.sort_values(["date", "id"]).copy()
        sp_cumul["cumul"] = sp_cumul["return_units"].cumsum()
        sp_color = SPORT_COLOR[sport.capitalize()]

        fig_sp = go.Figure(go.Scatter(
            x=sp_cumul["date"], y=sp_cumul["cumul"],
            mode="lines+markers",
            line=dict(color=sp_color, width=2.5),
            marker=dict(
                color=[GREEN if r == "win" else (RED if r == "loss" else MUTED)
                       for r in sp_cumul["result"]],
                size=7,
            ),
            hovertemplate="%{x|%d %b %Y}<br>Cumulative P&L: <b>%{y:+.2f}u</b><extra></extra>",
        ))
        fig_sp.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig_sp.update_layout(
            plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
            font=dict(color=MUTED),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Cumulative Units", tickformat="+.2f"),
            margin=dict(l=0, r=0, t=12, b=0), height=260,
        )
        st.plotly_chart(fig_sp, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            bt = (sp.groupby("bet_type_clean")["return_units"]
                    .agg(["sum", "count"]).reset_index()
                    .rename(columns={"bet_type_clean": "Bet Type", "sum": "P&L", "count": "Bets"})
                    .sort_values("P&L", ascending=False))
            st.plotly_chart(bar_chart(bt, "Bet Type", "P&L", "P&L by Bet Type"), use_container_width=True)

        with col_b:
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow = (sp.groupby("dow")["return_units"].sum()
                     .reindex(dow_order).fillna(0).reset_index()
                     .rename(columns={"dow": "Day", "return_units": "P&L"}))
            st.plotly_chart(bar_chart(dow, "Day", "P&L", "P&L by Day of Week"), use_container_width=True)

        if sport == "soccer":
            lg = (sp.groupby("league_clean")["return_units"]
                    .agg(["sum", "count"]).reset_index()
                    .rename(columns={"league_clean": "Competition", "sum": "P&L", "count": "Bets"})
                    .sort_values("P&L", ascending=False))
            st.plotly_chart(bar_chart(lg, "Competition", "P&L", "P&L by Competition"), use_container_width=True)

        stake_grp = (sp.groupby("units")["return_units"]
                       .agg(["sum", "count"]).reset_index()
                       .rename(columns={"units": "Stake", "sum": "P&L", "count": "Bets"})
                       .sort_values("Stake"))
        stake_grp["Stake"] = stake_grp["Stake"].map(lambda x: f"{x}u")
        st.plotly_chart(bar_chart(stake_grp, "Stake", "P&L", "P&L by Stake Size"), use_container_width=True)

st.markdown("---")

# ── Bankroll Compound Calculator ───────────────────────────────────────────────
st.markdown('<div class="section-title">Bankroll Compound Calculator</div>', unsafe_allow_html=True)
st.markdown(
    f"<span style='color:{MUTED};font-size:0.85rem'>"
    f"See how your bankroll would have grown by following our picks with compounding — "
    f"profits reinvested monthly at your chosen unit size.</span>",
    unsafe_allow_html=True,
)
st.markdown("")

calc_tabs = st.tabs(["⚽  Soccer", "🎾  Tennis", "🏒  Hockey", "📊  Combined"])

for ctab, sport in zip(calc_tabs[:3], ["soccer", "tennis", "hockey"]):
    with ctab:
        d = SPORT_DEFAULTS[sport]
        render_compound_tab(sport, monthly, d["bankroll"], d["unit_pct"])

with calc_tabs[3]:
    st.markdown(f"<span style='color:{MUTED};font-size:0.85rem'>Select the sports you want to combine. Units from all selected sports are pooled together.</span>", unsafe_allow_html=True)
    sport_opts = st.multiselect(
        "Sports to include",
        options=["Soccer", "Tennis", "Hockey"],
        default=["Soccer", "Tennis", "Hockey"],
        key="combined_sports",
    )
    selected = [s.lower() for s in sport_opts]
    if selected:
        render_compound_tab(None, monthly, 20000, 0.05, selected_sports=selected)
    else:
        st.info("Select at least one sport.")

st.markdown("---")

# ── All Bets History ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">All Bets History</div>', unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns([2, 2, 2])
sport_filter  = fc1.multiselect("Sport",   ["Soccer", "Tennis", "Hockey"],
                                 default=["Soccer", "Tennis", "Hockey"])
result_filter = fc2.multiselect("Result",  ["Win", "Loss", "Push", "Pending"],
                                 default=["Win", "Loss", "Push", "Pending"])
bet_filter    = fc3.multiselect("Bet Type", sorted(df_all["bet_type_clean"].unique()))

df_table = df_all[df_all["sport_label"].isin(sport_filter)].copy()
df_table = df_table[df_table["result"].str.capitalize().isin(result_filter)]
if bet_filter:
    df_table = df_table[df_table["bet_type_clean"].isin(bet_filter)]

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
display["units"]  = display["units"].map(lambda x: f"{x}u")

display.columns = ["Date", "Sport", "Competition", "Pick", "Stake", "Bet Type", "Odds", "Result", "P&L"]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    height=520,
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
    }
)

st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:0.75rem;margin-top:24px'>"
    f"Le Top Paddock · All records verified · Bet responsibly</div>",
    unsafe_allow_html=True,
)
