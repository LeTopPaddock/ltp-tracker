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
      border-radius: 12px; padding: 22px 16px; text-align: center;
  }}
  .metric-value {{ font-size: 1.9rem; font-weight: 700; margin-bottom: 4px; }}
  .metric-label {{ font-size: 0.78rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}
  .section-title {{
      font-size: 1.1rem; font-weight: 600; color: {TEXT};
      border-left: 3px solid {GOLD}; padding-left: 10px; margin: 24px 0 12px;
  }}
  .sport-pill {{
      display:inline-block; border-radius:6px; padding:3px 10px;
      font-size:0.8rem; font-weight:600; margin-right:6px;
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
    "btts":          "BTTS",
    "total":         "Over / Under",
    "goals o/u":     "Over / Under",
    "to draw":       "Draw",
    "to win":        "Moneyline",
    "moneyline":     "Moneyline",
    "handicap":      "Handicap",
    "shots":         "Player Prop",
    "player assist": "Player Prop",
}

LEAGUE_MAP = {
    "soccer":             "Various (Multi-League Parlays)",
    "OtherEnglish":       "EFL (Championship / League 1&2)",
    "Champions":          "UEFA Champions League",
    "Europa":             "UEFA Europa League",
    "Conference League":  "UEFA Conference League",
    "Spain":              "La Liga",
    "Italy":              "Serie A",
    "Germany":            "Bundesliga",
    "France":             "Ligue 1",
    "England":            "Premier League",
    "Argentina":          "Argentine Primera",
    "Denmark":            "Danish Superliga",
    "nhl":                "NHL",
    "atp 500":            "ATP 500",
    "atp 250":            "ATP 250",
    "challenger":         "ATP Challenger",
}

SPORT_COLOR = {"Soccer": GOLD, "Tennis": "#4B9EFF", "Hockey": "#A8D8EA"}
SPORT_EMOJI = {"Soccer": "⚽", "Tennis": "🎾", "Hockey": "🏒"}


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=180)
def load() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM picks WHERE result != 'pending' ORDER BY date, id",
        conn, parse_dates=["date"],
    )
    conn.close()

    df["sport_label"]    = df["sport"].str.capitalize()
    df["bet_type_clean"] = df["bet_type"].map(lambda x: BET_TYPE_MAP.get(str(x).lower(), str(x).capitalize() if x else "Other"))
    df["league_clean"]   = df["league"].map(lambda x: LEAGUE_MAP.get(str(x), str(x).title() if x else "Unknown"))
    df["dow"]            = df["date"].dt.day_name()
    df["month"]          = df["date"].dt.to_period("M").astype(str)
    return df


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
    total  = len(df)
    staked = df["units"].sum()
    profit = df["return_units"].sum()
    roi    = profit / staked * 100 if staked else 0
    wr     = wins / (wins + losses) * 100 if (wins + losses) else 0
    avg_o  = df[df["odds"].notna()]["odds"].mean() if df["odds"].notna().any() else 0
    return dict(total=total, wins=wins, losses=losses, pushes=pushes,
                win_rate=wr, units_staked=staked, units_profit=profit,
                roi=roi, avg_odds=avg_o)


def bar_chart(df_in, x_col, y_col, title, color_positive=True):
    """Bar chart — green for positive values, red for negative. No color for neutral."""
    colors = [GREEN if v >= 0 else RED for v in df_in[y_col]]
    fig = go.Figure(go.Bar(
        x=df_in[x_col], y=df_in[y_col],
        marker_color=colors,
        hovertemplate="%{x}<br>%{y:+.2f}u<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=13)),
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=MUTED),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=True,
                   zerolinecolor="#3a3f52", zerolinewidth=1),
        margin=dict(l=0, r=0, t=36, b=0),
        height=280,
    )
    return fig


def line_chart(df_in, x, y, color=None, title="", height=360):
    fig = px.line(
        df_in, x=x, y=y, color=color,
        color_discrete_sequence=[GOLD, "#4B9EFF", "#A8D8EA"],
        template="plotly_dark",
    )
    fig.update_traces(line=dict(width=2.5))
    fig.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=13)),
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=MUTED),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Units"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        margin=dict(l=0, r=0, t=36, b=0),
        height=height,
        hovermode="x unified",
    )
    return fig


def result_icon(r):
    return {"win": "✅ Win", "loss": "✖ Loss", "push": "➖ Push"}.get(r, r)


# ── App ────────────────────────────────────────────────────────────────────────
df = load()

# Header
st.markdown(f"# 🏆  Le Top Paddock")
st.markdown(f"<span style='color:{MUTED};font-size:0.9rem'>Verified picks record · Updated live</span>", unsafe_allow_html=True)

if df.empty:
    st.info("No picks data found. Run ltp_migrate.py first.")
    st.stop()

s = sport_stats(df)

# ── Overall stat row ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Overall Record — All Sports</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
stat_card(c1, f"{s['wins']}W / {s['losses']}L", "Win / Loss")
stat_card(c2, f"{s['win_rate']:.1f}%",   "Win Rate",
          GREEN if s['win_rate'] >= 55 else (MUTED if s['win_rate'] >= 50 else RED))
stat_card(c3, f"{s['roi']:+.1f}%",       "ROI",
          GREEN if s['roi'] > 0 else RED)
stat_card(c4, f"{s['units_profit']:+.2f}u", "Total P&L",
          GREEN if s['units_profit'] > 0 else RED)
stat_card(c5, f"{s['avg_odds']:.2f}",    "Avg Odds")
stat_card(c6, str(s['total']),           "Total Bets")
stat_card(c7, f"{s['units_staked']:.1f}u", "Units Staked")

st.markdown("")

# ── P&L Chart ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Cumulative P&L by Sport</div>', unsafe_allow_html=True)

df_pl = df.sort_values(["sport", "date", "id"]).copy()
df_pl["cumul"] = df_pl.groupby("sport")["return_units"].cumsum()
df_pl["sport_label"] = df_pl["sport"].str.capitalize()

fig_pl = line_chart(df_pl, "date", "cumul", color="sport_label", title="", height=360)
st.plotly_chart(fig_pl, use_container_width=True)

# ── Sport overview cards ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">By Sport</div>', unsafe_allow_html=True)
cols = st.columns(3)

for col, sport in zip(cols, ["soccer", "tennis", "hockey"]):
    sp = df[df["sport"] == sport]
    ss = sport_stats(sp)
    em = SPORT_EMOJI[sport.capitalize()]
    sc = SPORT_COLOR[sport.capitalize()]

    with col:
        sign = "+" if ss["units_profit"] >= 0 else ""
        pl_color = GREEN if ss["units_profit"] > 0 else RED
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.3rem;font-weight:700;color:{sc};margin-bottom:12px">'
            f'{em}  {sport.capitalize()}</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
            f'<span style="color:{MUTED};font-size:0.8rem">Record</span>'
            f'<span style="font-weight:600">{ss["wins"]}W / {ss["losses"]}L</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
            f'<span style="color:{MUTED};font-size:0.8rem">Win Rate</span>'
            f'<span style="font-weight:600">{ss["win_rate"]:.1f}%</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
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
tabs = st.tabs(["⚽  Soccer", "🎾  Tennis", "🏒  Hockey"])

for tab, sport in zip(tabs, ["soccer", "tennis", "hockey"]):
    with tab:
        sp = df[df["sport"] == sport].copy()
        if sp.empty:
            st.info(f"No {sport} picks yet.")
            continue

        ss = sport_stats(sp)

        # Stat row
        r1,r2,r3,r4,r5 = st.columns(5)
        stat_card(r1, f"{ss['wins']}W / {ss['losses']}L", "Record")
        stat_card(r2, f"{ss['win_rate']:.1f}%", "Win Rate",
                  GREEN if ss["win_rate"] >= 55 else (MUTED if ss["win_rate"] >= 50 else RED))
        stat_card(r3, f"{ss['roi']:+.1f}%", "ROI", GREEN if ss["roi"] > 0 else RED)
        stat_card(r4, f"{ss['units_profit']:+.2f}u", "P&L",
                  GREEN if ss["units_profit"] > 0 else RED)
        stat_card(r5, f"{ss['avg_odds']:.2f}", "Avg Odds")

        st.markdown("")

        # P&L line
        sp_cumul = sp.sort_values(["date","id"]).copy()
        sp_cumul["cumul"] = sp_cumul["return_units"].cumsum()
        fig_sp = go.Figure(go.Scatter(
            x=sp_cumul["date"], y=sp_cumul["cumul"],
            mode="lines+markers",
            line=dict(color=SPORT_COLOR[sport.capitalize()], width=2.5),
            marker=dict(
                color=[GREEN if r == "win" else (RED if r == "loss" else MUTED)
                       for r in sp_cumul["result"]],
                size=7,
            ),
            hovertemplate="%{x|%d %b}<br>Cumulative: %{y:+.2f}u<extra></extra>",
        ))
        fig_sp.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig_sp.update_layout(
            plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
            font=dict(color=MUTED),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title=""),
            yaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Cumulative Units"),
            margin=dict(l=0, r=0, t=12, b=0), height=280,
        )
        st.plotly_chart(fig_sp, use_container_width=True)

        col_a, col_b = st.columns(2)

        # By bet type
        with col_a:
            bt = (sp.groupby("bet_type_clean")["return_units"]
                    .agg(["sum","count"]).reset_index()
                    .rename(columns={"bet_type_clean":"Bet Type","sum":"P&L","count":"Count"})
                    .sort_values("P&L", ascending=False))
            st.plotly_chart(bar_chart(bt, "Bet Type", "P&L", "P&L by Bet Type"), use_container_width=True)

        # By day of week
        with col_b:
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow = (sp.groupby("dow")["return_units"].sum()
                     .reindex(dow_order).fillna(0).reset_index()
                     .rename(columns={"dow":"Day","return_units":"P&L"}))
            st.plotly_chart(bar_chart(dow, "Day", "P&L", "P&L by Day of Week"), use_container_width=True)

        # By league (soccer only — too noisy for tennis/hockey)
        if sport == "soccer":
            lg = (sp.groupby("league_clean")["return_units"]
                    .agg(["sum","count"]).reset_index()
                    .rename(columns={"league_clean":"League","sum":"P&L","count":"Count"})
                    .sort_values("P&L", ascending=False))
            st.plotly_chart(bar_chart(lg, "League", "P&L", "P&L by Competition"), use_container_width=True)

        # By stake size
        stake_grp = (sp.groupby("units")["return_units"]
                       .agg(["sum","count"]).reset_index()
                       .rename(columns={"units":"Stake","sum":"P&L","count":"Count"})
                       .sort_values("Stake"))
        stake_grp["Stake"] = stake_grp["Stake"].map(lambda x: f"{x}u")
        st.plotly_chart(bar_chart(stake_grp, "Stake", "P&L", "P&L by Stake Size"), use_container_width=True)

st.markdown("---")

# ── Full Bets Table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">All Bets History</div>', unsafe_allow_html=True)

# Filters
fc1, fc2, fc3 = st.columns([2,2,2])
sport_filter  = fc1.multiselect("Sport", ["Soccer","Tennis","Hockey"], default=["Soccer","Tennis","Hockey"])
result_filter = fc2.multiselect("Result", ["Win","Loss","Push"], default=["Win","Loss","Push"])
bet_filter    = fc3.multiselect("Bet Type", sorted(df["bet_type_clean"].unique()))

df_table = df[df["sport_label"].isin(sport_filter)].copy()
df_table = df_table[df_table["result"].str.capitalize().isin(result_filter)]
if bet_filter:
    df_table = df_table[df_table["bet_type_clean"].isin(bet_filter)]

# Build display table
display = df_table.sort_values("date", ascending=False)[[
    "date","sport_label","league_clean","description",
    "units","bet_type_clean","odds","result","return_units"
]].copy()

display["date"]         = display["date"].dt.strftime("%d %b %Y")
display["return_units"] = display["return_units"].map(lambda x: f"{x:+.2f}u")
display["odds"]         = display["odds"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
display["result"]       = display["result"].map(result_icon)
display["units"]        = display["units"].map(lambda x: f"{x}u")

display.columns = ["Date","Sport","Competition","Pick","Stake","Bet Type","Odds","Result","P&L"]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    height=500,
    column_config={
        "P&L": st.column_config.TextColumn("P&L", width="small"),
        "Stake": st.column_config.TextColumn("Stake", width="small"),
        "Odds": st.column_config.TextColumn("Odds", width="small"),
        "Result": st.column_config.TextColumn("Result", width="small"),
        "Pick": st.column_config.TextColumn("Pick", width="large"),
    }
)

st.markdown(f"<div style='text-align:center;color:{MUTED};font-size:0.75rem;margin-top:24px'>Le Top Paddock · All records verified · Bet responsibly</div>", unsafe_allow_html=True)
