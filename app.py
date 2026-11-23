import akshare as ak
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------- 页面基础设置 ----------------
st.set_page_config(page_title="指数估值看板", page_icon="📊", layout="wide")

# ---------------- 指数池：想增删指数，直接改这个字典 ----------------
INDEX_POOL = {
    "000300": "沪深300",
    "000016": "上证50",
    "000905": "中证500",
    "000852": "中证1000",
    "000922": "中证红利",
    "000015": "红利指数",
    "000932": "中证消费",
}

# ---------------- 数据获取（带缓存，1小时内不重复请求） ----------------
@st.cache_data(ttl=3600, show_spinner="正在从指数官网获取数据...")
def get_index_data(code: str) -> pd.DataFrame:
    # akshare 1.20.44 起，stock_zh_index_value_csindex 更名为 index_value_csindex
    try:
        try:
            df = ak.index_value_csindex(symbol=code)        # 新版接口
        except TypeError:
            df = ak.index_value_csindex(code)               # 参数名变了时兜底
    except AttributeError:
        df = ak.stock_zh_index_value_csindex(symbol=code)   # 旧版接口兜底

    df.columns = [str(c) for c in df.columns]

    # 自动定位「日期」列
    date_col = next((c for c in ["trade_date", "date", "日期", "交易日"] if c in df.columns), None)
    # 自动定位「市盈率」列
    pe_col = next((c for c in ["pe_ttm", "pe1", "pe2", "市盈率1", "市盈率", "市盈率TTM"] if c in df.columns), None)

    if date_col is None or pe_col is None:
        raise ValueError(f"列名不匹配，实际列名为：{list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "trade_date", pe_col: "pe_ttm"})
    return df.sort_values("trade_date").dropna(subset=["pe_ttm"]).reset_index(drop=True)

def get_bond_series() -> pd.Series:
    """返回「日期 → 10年期国债收益率(小数)」的时间序列"""
    # akshare 1.20.41 起，bond_zh_us_rate 更名为 bond_zh_us_yield_em
    try:
        bond = ak.bond_zh_us_yield_em()                    # 新版接口
    except AttributeError:
        bond = ak.bond_zh_us_rate(start_date="20150101")   # 旧版兜底

    bond.columns = [str(c) for c in bond.columns]
    candidates = [c for c in bond.columns if "10" in c and "国债" in c] or \
                 [c for c in bond.columns if "10" in c]
    if not candidates:
        raise ValueError(f"未找到10年期国债收益率列，实际列名为：{list(bond.columns)}")
    date_col = next((c for c in ["日期", "date", "trade_date"] if c in bond.columns), bond.columns[0])

    s = bond[[date_col, candidates[0]]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.set_index(date_col)[candidates[0]].astype(float).sort_index()
    if s.iloc[-1] > 0.5:   # 单位自适应：百分数 → 小数
        s = s / 100
    return s

@st.cache_data(ttl=3600, show_spinner="正在获取国债收益率...")
def get_r10() -> float:
    # akshare 1.20.41 起，bond_zh_us_rate 更名为 bond_zh_us_yield_em
    try:
        bond = ak.bond_zh_us_yield_em()                    # 新版接口
    except AttributeError:
        bond = ak.bond_zh_us_rate(start_date="20150101")   # 旧版兜底

    bond.columns = [str(c) for c in bond.columns]
    # 自动匹配「10年期国债收益率」列（不同版本列名可能略有差异）
    candidates = [c for c in bond.columns if "10" in c and "国债" in c] or \
                 [c for c in bond.columns if "10" in c]
    if not candidates:
        raise ValueError(f"未找到10年期国债收益率列，实际列名为：{list(bond.columns)}")

    val = float(bond[candidates[0]].dropna().iloc[-1])
    return val / 100 if val > 0.5 else val   # 单位自适应（%或小数）

# ---------------- 指标计算 ----------------
def pct_rank(series: pd.Series, value: float) -> float:
    """当前值在历史序列中的百分位（0~100）"""
    return round((series < value).mean() * 100, 1)

def level(pct: float):
    """根据百分位返回（评级文字, 颜色）"""
    if pct < 30:
        return "🟢 低估", "#43a047"
    if pct < 70:
        return "🟡 正常", "#fdd835"
    return "🔴 高估", "#e53935"

def get_metrics(code: str, years: int, r10: float) -> dict:
    df = get_index_data(code)
    today = df.iloc[-1]
    sub = df[df["trade_date"] >= today["trade_date"] - pd.DateOffset(years=years)]
    pe = float(today["pe_ttm"])
    earn = 1 / pe                          # 盈利收益率 = 1 / PE
    return {
        "code": code,
        "指数": INDEX_POOL[code],
        "数据日期": today["trade_date"].strftime("%Y-%m-%d"),
        "PE-TTM": round(pe, 2),
        "百分位": pct_rank(sub["pe_ttm"], pe),
        "盈利收益率": earn,
        "股债利差": earn - r10,
        "格雷厄姆指数": earn / r10,
    }
@st.cache_data(ttl=3600, show_spinner=False)
def get_spread_history(code: str, years: int, bond: pd.Series) -> pd.DataFrame:
    """计算某指数近 N 年每个交易日的股债利差"""
    df = get_index_data(code)
    today = df.iloc[-1]["trade_date"]
    sub = df[df["trade_date"] >= today - pd.DateOffset(years=years)][["trade_date", "pe_ttm"]].copy()
    sub = sub.set_index("trade_date")
    sub["盈利收益率"] = 1 / sub["pe_ttm"]

    # 把国债收益率按日期对齐（指数有数据但国债缺数据的日子，用最近一天的值填充）
    all_idx = sub.index.union(bond.index)
    sub["r10"] = bond.reindex(all_idx).ffill().reindex(sub.index)

    sub = sub.dropna(subset=["r10"])
    sub["股债利差"] = sub["盈利收益率"] - sub["r10"]
    return sub.reset_index()

def spread_chart(hist: pd.DataFrame) -> go.Figure:
    """股债利差历史折线图"""
    cur = float(hist["股债利差"].iloc[-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["trade_date"],
        y=hist["股债利差"],
        mode="lines",
        line=dict(color="#1e88e5", width=2),
        fill="tozeroy",
        fillcolor="rgba(30,136,229,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>股债利差：%{y:.2%}<extra></extra>",
    ))
    # 最新值高亮标记
    fig.add_trace(go.Scatter(
        x=[hist["trade_date"].iloc[-1]],
        y=[cur],
        mode="markers+text",
        marker=dict(color="#e53935", size=8),
        text=[f"{cur:.2%}"],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#9e9e9e",
                  annotation_text="利差=0（股债性价比相同）",
                  annotation_font_size=10, annotation_font_color="#9e9e9e")
    fig.update_layout(
        height=320,
        margin=dict(t=15, b=0, l=10, r=10),
        yaxis_tickformat=".1%",
        hovermode="x unified",
    )
    return fig

# ---------------- 温度计绘图 ----------------
def gauge(name: str, pct: float):
    _, color = level(pct)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 30}},
        title={"text": name, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#c8e6c9"},   # 绿色区：低估
                {"range": [30, 70], "color": "#fff9c4"},  # 黄色区：正常
                {"range": [70, 100], "color": "#ef9a9a"}, # 红色区：高估
            ],
        },
    ))
    fig.update_layout(height=230, margin=dict(t=40, b=0, l=25, r=25))
    return fig

# ================== 页面主体 ==================
st.title("📊 指数估值看板")
st.caption("数据来源：中证指数官网 | 缓存1小时，可在左侧强制刷新")

# ---------------- 侧边栏：增删指数在这里操作 ----------------
with st.sidebar:
    st.header("⚙️ 设置")
    selected = st.multiselect(
        "勾选 / 取消勾选 = 加入 / 删除指数",
        options=list(INDEX_POOL.keys()),
        format_func=lambda c: f"{INDEX_POOL[c]}（{c}）",
        default=list(INDEX_POOL.keys()),
    )
    years = st.select_slider("百分位统计区间", options=[3, 5, 10], value=5)
    if st.button("🔄 强制刷新数据"):
        st.cache_data.clear()
        st.rerun()

if not selected:
    st.info("👈 请在左侧选择至少一个指数")
    st.stop()

bond_series = get_bond_series()
r10 = float(bond_series.iloc[-1])


# ---------------- 逐个计算指标 ----------------
metrics_list, errors = [], []
for code in selected:
    try:
        metrics_list.append(get_metrics(code, years, r10))
    except Exception as e:
        errors.append(f"{INDEX_POOL[code]}：{e}")

for msg in errors:
    st.warning(f"⚠️ 数据获取失败 —— {msg}")

if not metrics_list:
    st.stop()

# ---------------- 顶部：整体股债性价比 ----------------
avg_graham = sum(m["格雷厄姆指数"] for m in metrics_list) / len(metrics_list)
c1, c2, c3 = st.columns(3)
c1.metric("10 年期国债收益率", f"{r10:.2%}")
c2.metric("格雷厄姆指数（平均）", f"{avg_graham:.2f}", help="大于2说明股票相对债券更值得买")
c3.metric("整体评价", "🟢 股优于债" if avg_graham > 2 else "🔴 债优于股")

st.divider()

# ---------------- 汇总表格 ----------------
st.subheader("📋 指标汇总")
pct_col = f"百分位({years}年)"
table = pd.DataFrame([
    {
        "指数": m["指数"],
        "数据日期": m["数据日期"],
        "PE-TTM": m["PE-TTM"],
        pct_col: m["百分位"],
        "评级": level(m["百分位"])[0],
        "盈利收益率": f'{m["盈利收益率"]:.2%}',
        "股债利差": f'{m["股债利差"]:.2%}',
        "格雷厄姆指数": round(m["格雷厄姆指数"], 2),
    }
    for m in metrics_list
]).sort_values(pct_col)
st.dataframe(table, use_container_width=True, hide_index=True)
st.caption("百分位越低 = 当前估值在历史上越便宜；股债利差 = 盈利收益率 − 10Y国债收益率")

# ---------------- 温度计 ----------------
st.subheader("🌡️ 估值温度计")
for i in range(0, len(metrics_list), 2):
    pair = metrics_list[i : i + 2]
    cols = st.columns(len(pair))
    for col, m in zip(cols, pair):
        with col:
            st.plotly_chart(gauge(m["指数"], m["百分位"]), use_container_width=True)

# ---------------- 股债利差历史走势 ----------------
st.divider()
st.subheader(f"📈 股债利差走势（近{years}年）")
st.caption("股债利差 = 盈利收益率(1/PE) − 10年期国债收益率，数值越高 = 股票相对债券越值得买；左上角可切换指数")

tabs = st.tabs([m["指数"] for m in metrics_list])
for tab, m in zip(tabs, metrics_list):
    with tab:
        try:
            hist = get_spread_history(m["code"], years, bond_series)
            st.caption(f"最新：{hist['trade_date'].iloc[-1].strftime('%Y-%m-%d')} ｜ 股债利差 {hist['股债利差'].iloc[-1]:.2%}")
            st.plotly_chart(spread_chart(hist), use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ {m['指数']} 走势绘制失败：{e}")
