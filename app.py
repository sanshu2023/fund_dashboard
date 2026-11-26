import akshare as ak
import inspect
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
# ---------- 数据源①：中证指数官网（主源，显式传入日期区间） ----------
def _fetch_csi(code: str) -> pd.DataFrame:
    if hasattr(ak, "index_value_csindex"):          # 1.20.44+ 新版
        func = ak.index_value_csindex
    else:                                           # 旧版兜底
        func = ak.stock_zh_index_value_csindex

    # 自动探测接口支持的日期参数，显式要「成立以来」全部数据
    kwargs = {}
    params = inspect.signature(func).parameters
    if "start_date" in params:
        kwargs["start_date"] = "20040101"           # 早于所有指数成立日
    if "end_date" in params:
        kwargs["end_date"] = pd.Timestamp.today().strftime("%Y%m%d")

    try:
        return func(symbol=code, **kwargs)
    except TypeError:
        return func(code, **kwargs)
    
# ---------- 数据源②：乐咕乐股（备用） ----------
LG_FUNC_NAMES = ["index_pe_lg", "stock_index_pe_lg"]   # 新旧版本函数名都试
LG_NAME_MAP = {
    "000016": "上证50", "000300": "沪深300", "000905": "中证500",
    "000015": "上证红利", "399006": "创业板指", "000852": "中证1000",
    "000688": "科创50", "000922": "中证红利",
}

def _fetch_lg(code: str) -> pd.DataFrame:
    if code not in LG_NAME_MAP:
        raise RuntimeError("该数据源不支持此指数")
    last_err = None
    for fname in LG_FUNC_NAMES:
        func = getattr(ak, fname, None)
        if func is None:
            continue
        try:
            return func(symbol=LG_NAME_MAP[code])
        except Exception as e:
            last_err = e
    raise RuntimeError(f"不可用：{last_err}")

# ---------- 数据源③：韭圈儿（备用，覆盖主题指数） ----------
FUNDDB_NAME_MAP = {
    "000922": "中证红利", "000015": "上证红利", "000932": "中证消费",
    "000300": "沪深300", "000905": "中证500", "000016": "上证50",
    "399006": "创业板指", "000852": "中证1000", "000688": "科创50",
}

def _fetch_funddb(code: str) -> pd.DataFrame:
    if code not in FUNDDB_NAME_MAP:
        raise RuntimeError("该数据源不支持此指数")
    func = getattr(ak, "index_value_hist_funddb", None)
    if func is None:
        raise RuntimeError("当前 akshare 版本无此接口")
    try:
        return func(symbol=FUNDDB_NAME_MAP[code], indicator="市盈率")
    except TypeError:
        return func(FUNDDB_NAME_MAP[code], "市盈率")


# ---------- 列名统一 + 主函数 ----------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c) for c in df.columns]
    date_col = next((c for c in ["trade_date", "date", "日期", "交易日"] if c in df.columns), None)
    pe_col = next((c for c in ["pe_ttm", "pe1", "市盈率", "市盈率TTM"] if c in df.columns), None)
    if pe_col is None:
        pe_col = next((c for c in df.columns if "pe" in c.lower() or "市盈率" in c), None)
    if date_col is None or pe_col is None:
        raise ValueError(f"列名不匹配，实际列名为：{list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "trade_date", pe_col: "pe_ttm"})
    return df.sort_values("trade_date").dropna(subset=["pe_ttm"]).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner="正在获取指数估值历史...")
def get_index_data(code: str) -> pd.DataFrame:
    """数据源依次尝试：中证官网(带日期区间) → 乐咕乐股 → 韭圈儿 → 明确报错"""
    errors = []
    for fetcher in (_fetch_csi, _fetch_lg, _fetch_funddb):
        try:
            df = _normalize(fetcher(code))
            span = (df["trade_date"].iloc[-1] - df["trade_date"].iloc[0]).days / 365.25
            if span >= 2:            # 历史不足2年视为不合格，换下一个源
                return df
            errors.append(f"{fetcher.__name__} 仅返回 {span:.1f} 年数据")
        except Exception as e:
            errors.append(f"{fetcher.__name__}：{e}")
    raise RuntimeError("；".join(errors))

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
@st.cache_data(ttl=3600, show_spinner=False)
def get_spread_history(code: str, bond: pd.Series) -> pd.DataFrame:
    """计算某指数成立以来每个交易日的股债利差及其历史百分位"""
    df = get_index_data(code)
    sub = df[["trade_date", "pe_ttm"]].set_index("trade_date").copy()
    sub["盈利收益率"] = 1 / sub["pe_ttm"]

    # 国债收益率按日期对齐，缺失日沿用最近一天
    all_idx = sub.index.union(bond.index)
    sub["r10"] = bond.reindex(all_idx).ffill().reindex(sub.index)
    sub = sub.dropna(subset=["r10"])
    sub["股债利差"] = sub["盈利收益率"] - sub["r10"]

    # 成立以来百分位：每天的利差在「截至当天」历史中的分位（不偷看未来数据）
    sub["百分位"] = sub["股债利差"].expanding(min_periods=2).rank(pct=True) * 100
    return sub.reset_index()

def spread_chart(hist: pd.DataFrame) -> go.Figure:
    """浅网格 + 柔和分段色 + 线端数值 + 图例置于图下方横排"""
    cur = float(hist["股债利差"].iloc[-1])
    cur_pct = float(hist["百分位"].iloc[-1])
    end_date = hist["trade_date"].iloc[-1]

    def color_for(p):
        if pd.isna(p): return "#9aa5b1"
        if p >= 75:    return "#43a047"   # 利差高位 → 便宜 → 绿
        if p <= 25:    return "#e53935"   # 利差低位 → 贵 → 红
        return "#1e88e5"                  # 中性 → 蓝

    hist = hist.copy()
    hist["颜色"] = hist["百分位"].apply(color_for)

    fig = go.Figure()
    start, n = 0, len(hist)
    for i in range(1, n + 1):
        if i == n or hist["颜色"].iloc[i] != hist["颜色"].iloc[start]:
            seg = hist.iloc[max(start - 1, 0): i]
            fig.add_trace(go.Scatter(
                x=seg["trade_date"], y=seg["股债利差"], mode="lines",
                line=dict(color=hist["颜色"].iloc[start], width=1.8),
                customdata=seg[["百分位"]].values,
                hovertemplate="%{x|%Y-%m-%d}<br>股债利差：%{y:.2%}<br>成立以来百分位：%{customdata[0]:.0f}%<extra></extra>",
                showlegend=False,
            ))
            start = i

    # 线端最新值标签
    fig.add_annotation(
        x=end_date, y=cur, xanchor="left", xshift=4,
        text=f"<b>{cur:.2%}</b><br><span style='font-size:10px'>{cur_pct:.0f}%分位</span>",
        showarrow=False, font=dict(color=color_for(cur_pct), size=11),
    )

    # 横轴：按跨度自动均分刻度
    span_years = (hist["trade_date"].iloc[-1] - hist["trade_date"].iloc[0]).days / 365.25
    step = min([0.5, 1, 2, 5, 10, 20], key=lambda v: abs(v - span_years / 8))

    fig.update_xaxes(
        dtick=f"M{int(step * 12)}", showgrid=False, linecolor="#cfd8dc",
        tickfont=dict(color="#78909c", size=11),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(count=3, label="3年", step="year", stepmode="backward"),
                dict(count=5, label="5年", step="year", stepmode="backward"),
                dict(step="all", label="全部"),
            ],
            font=dict(size=11, color="#546e7a"),
            bgcolor="#f5f7fa", activecolor="#dbeafe",
            bordercolor="#cfd8dc", borderwidth=1,
        ),
    )
    fig.update_yaxes(
        tickformat=".0%", zeroline=False,
        gridcolor="#edeef2", griddash="dot", gridwidth=1,
        tickfont=dict(color="#78909c", size=11),
    )
    fig.update_layout(
        height=380, hovermode="x unified",
        margin=dict(t=40, b=60, l=10, r=42),   # 底部留白给图例，右侧仅留线端标签空间
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",            # 水平排列
            yanchor="bottom", y=-0.22,  # 移到图表下方
            xanchor="left", x=0,
            font=dict(size=11, color="#546e7a"),
            itemsizing="constant",
        ),
    )

    # 图例（短文案，适配手机横排）
    for label, c in [("高位 = 便宜", "#43a047"), ("中性", "#1e88e5"), ("低位 = 贵", "#e53935")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=c, size=9, symbol="square"), name=label,
        ))
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
st.subheader("📈 股债利差走势（指数成立以来）")
st.caption("股债利差 = 盈利收益率(1/PE) − 10年期国债收益率 ｜ 🟢 利差高位＝相对便宜 · 🔵 中性 · 🔴 利差低位＝相对昂贵")

tabs = st.tabs([m["指数"] for m in metrics_list])
for tab, m in zip(tabs, metrics_list):
    with tab:
        try:
            hist = get_spread_history(m["code"], bond_series)
            cur_pct = hist["百分位"].iloc[-1]
            st.caption(
                f"最新：{hist['trade_date'].iloc[-1].strftime('%Y-%m-%d')} ｜ "
                f"股债利差 {hist['股债利差'].iloc[-1]:.2%} ｜ "
                f"成立以来百分位 {cur_pct:.0f}%"
            )
            st.plotly_chart(spread_chart(hist), use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ {m['指数']} 走势绘制失败：{e}")
