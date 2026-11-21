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
    df = ak.stock_zh_index_value_csindex(symbol=code)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").dropna(subset=["pe_ttm"]).reset_index(drop=True)

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

r10 = get_r10()

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
table = pd.DataFrame([
    {
        "指数": m["指数"],
        "数据日期": m["数据日期"],
        "PE-TTM": m["PE-TTM"],
        f"百分位({years}年)": m["百分位"],
        "评级": level(m["百分位"])[0],
        "盈利收益率": f'{m["盈利收益率"]:.2%}',
        "股债利差": f'{m["股债利差"]:.2%}',
        "格雷厄姆指数": round(m["格雷厄姆指数"], 2),
    }
    for m in metrics_list
]).sort_values("百分位")
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
