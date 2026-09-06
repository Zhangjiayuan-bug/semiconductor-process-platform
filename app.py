from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.analysis import (
    capability,
    clean_data,
    detect_anomalies,
    drift_report,
    feature_importance,
    load_secom,
    secom_report,
    secom_sensor_association,
    secom_validation_crosscheck,
    spc_chart,
    validate_data,
)

ROOT = Path(__file__).parent


@st.cache_data(show_spinner=False)
def cached_load_secom(path: str) -> pd.DataFrame:
    data, _ = load_secom(path)
    return data


@st.cache_data(show_spinner=False)
def cached_secom_report(data: pd.DataFrame) -> dict[str, object]:
    return secom_report(data)


@st.cache_data(show_spinner=False)
def cached_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return drift_report(reference, current, columns=columns)


@st.cache_data(show_spinner=False)
def cached_sensor_association(data: pd.DataFrame) -> pd.DataFrame:
    return secom_sensor_association(data)


@st.cache_data(show_spinner=False)
def cached_validation_crosscheck(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return secom_validation_crosscheck(data)


st.set_page_config(page_title="半导体工艺数据分析平台", page_icon="◈", layout="wide")
st.title("半导体工艺数据分析平台")
st.caption("面向 PE、PIE 与 EE 场景的工艺监控、异常筛查和质量分析")

mode = st.sidebar.radio("数据模式", ["UCI SECOM 真实数据", "工艺质量演示模板"])
if mode == "UCI SECOM 真实数据":
    secom_dir = ROOT / "data" / "raw" / "uci_secom"
    try:
        data = cached_load_secom(str(secom_dir))
    except Exception as exc:
        st.error(f"无法载入 SECOM 数据：{exc}")
        st.stop()
    report = cached_secom_report(data)
    quality = report["quality"]
    st.info("当前使用 UCI SECOM 真实公开数据集（CC BY 4.0）。原始标签中 -1 表示通过、1 表示失效；传感器名称已由数据源匿名化。")
    cards = st.columns(5)
    cards[0].metric("样本数", f"{quality['rows']:,}")
    cards[1].metric("传感器数", f"{quality['sensors']:,}")
    cards[2].metric("失效样本", f"{quality['failures']:,}")
    cards[3].metric("失效率", f"{quality['failures'] / quality['rows'] * 100:.2f}%")
    cards[4].metric("传感器缺失率", f"{quality['missing_rate_pct']:.1f}%")
    overview, sensors, association_tab, drift_tab, model_tab = st.tabs(["数据质量", "传感器分析", "失效关联", "漂移监控", "失效预测"])
    with overview:
        left, right = st.columns(2)
        with left:
            st.subheader("类别分布")
            st.plotly_chart(
                px.bar(
                    pd.DataFrame({"结果": ["通过", "失效"], "数量": [(data.fail == 0).sum(), (data.fail == 1).sum()]}),
                    x="结果",
                    y="数量",
                    color="结果",
                    title="通过与失效样本数量",
                ),
                width="stretch",
            )
        with right:
            st.subheader("缺失率最高的传感器")
            missingness_view = (
                report["missingness"]
                .assign(missing_rate=lambda x: (x.missing_rate * 100).round(1))
                .rename(columns={"sensor": "传感器", "missing_rate": "缺失率（%）"})
            )
            st.dataframe(missingness_view, width="stretch", hide_index=True)
        st.caption("公开数据源未提供设备、批次、工艺步骤或连续测量规格字段，因此本模式不会虚构这些信息，也不计算 Cp/Cpk。")
    with sensors:
        sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
        selected = st.multiselect("选择传感器", sensor_cols, default=sensor_cols[:3])
        if selected:
            sensor_view = data[[selected[0], "fail"]].copy()
            sensor_view["结果"] = sensor_view.pop("fail").map({False: "通过", True: "失效"})
            st.plotly_chart(
                px.histogram(sensor_view, x=selected[0], color="结果", marginal="box", barmode="overlay", title=f"{selected[0]} 在通过/失效样本中的分布"),
                width="stretch",
            )
            sensor_table = data[selected + ["fail", "timestamp"]].head(100).rename(columns={"fail": "是否失效", "timestamp": "时间"})
            st.dataframe(sensor_table, width="stretch", hide_index=True)
    with association_tab:
        association = cached_sensor_association(data)
        st.subheader("传感器与失效标签的统计关联")
        st.caption(
            "使用 Welch t 检验比较通过组与失效组均值，并以 Benjamini-Hochberg 方法校正多重检验；标准化效应量用于衡量组间差异强度。统计关联不代表工艺因果。"
        )
        if association.empty:
            st.info("当前没有满足样本量和变异要求的传感器可用于关联筛查。")
        else:
            st.plotly_chart(
                px.bar(
                    association.sort_values("effect_size"),
                    x="effect_size",
                    y="sensor",
                    color="pvalue_bh",
                    orientation="h",
                    title="通过组与失效组的标准化均值差异",
                    labels={"effect_size": "标准化效应量", "sensor": "传感器", "pvalue_bh": "校正后 p 值"},
                ),
                width="stretch",
            )
            association_view = association.rename(
                columns={
                    "sensor": "传感器",
                    "pass_mean": "通过组均值",
                    "fail_mean": "失效组均值",
                    "mean_delta": "均值差",
                    "effect_size": "标准化效应量",
                    "welch_t": "Welch t 值",
                    "pvalue": "原始 p 值",
                    "pvalue_bh": "校正后 p 值",
                    "pass_count": "通过组样本数",
                    "fail_count": "失效组样本数",
                }
            )
            st.dataframe(association_view.round(6), width="stretch", hide_index=True)
            st.download_button("下载关联分析报告", association.to_csv(index=False), "SECOM_传感器失效关联.csv", "text/csv")
    with drift_tab:
        sensor_cols = [c for c in data.columns if c.startswith("sensor_")]
        ordered = data.sort_values("timestamp").reset_index(drop=True)
        split = int(len(ordered) * 0.7)
        drift = cached_drift_report(ordered.iloc[:split], ordered.iloc[split:], sensor_cols)
        st.subheader("按时间划分的传感器漂移")
        st.caption("按时间排序后，以前 70% 样本作为基准窗口、后 30% 作为当前窗口。PSI 与 KS 检验用于筛查分布变化，不代表已经定位工艺根因。")
        drift_counts = drift["status"].value_counts().reindex(["drift", "watch", "stable"], fill_value=0)
        drift_cards = st.columns(3)
        drift_cards[0].metric("明显漂移", f"{drift_counts['drift']:,}")
        drift_cards[1].metric("需要关注", f"{drift_counts['watch']:,}")
        drift_cards[2].metric("基本稳定", f"{drift_counts['stable']:,}")
        drift_view = drift.assign(状态=drift["status"].map({"drift": "明显漂移", "watch": "需要关注", "stable": "基本稳定"}))
        st.plotly_chart(
            px.bar(
                drift_view.head(20), x="psi", y="feature", color="状态", orientation="h", title="PSI 最高的传感器", labels={"psi": "PSI", "feature": "传感器"}
            ),
            width="stretch",
        )
        st.dataframe(drift.round(5), width="stretch", hide_index=True)
        st.download_button("下载漂移监控报告", drift.to_csv(index=False), "SECOM_传感器漂移报告.csv", "text/csv")
    with model_tab:
        st.subheader("防止数据泄漏的逻辑回归基线")
        st.caption("采用分层留出验证；中位数填补、缺失指示和标准化仅在训练流程内部拟合，并通过类别权重缓解失效样本稀少问题。")
        metrics_view = pd.DataFrame([report["metrics"]]).rename(
            columns={
                "balanced_accuracy": "平衡准确率",
                "precision": "精确率",
                "recall": "召回率",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "train_rows": "训练样本数",
                "test_rows": "测试样本数",
                "features_after_constant_filter": "过滤后特征数",
            }
        )
        st.dataframe(metrics_view.round(4), width="stretch", hide_index=True)
        predictions = report["predictions"]
        cm = confusion_matrix(predictions.actual_fail, predictions.predicted_fail)
        st.plotly_chart(
            px.imshow(cm, text_auto=True, x=["预测通过", "预测失效"], y=["实际通过", "实际失效"], color_continuous_scale="Blues", title="留出集混淆矩阵"),
            width="stretch",
        )
        st.download_button("下载留出集预测结果", predictions.to_csv(index=False), "SECOM_留出集预测.csv", "text/csv")
        st.divider()
        st.subheader("随机分层与时间外推交叉验证")
        st.caption(
            "交叉验证同时检验两种假设：5 折×2 次重复分层 CV 衡量近似 IID 场景，5 折扩展时间窗口衡量由历史预测未来的能力。所有预处理仅在各训练折内拟合，PR-AUC 为主指标。"
        )
        if st.button("运行交叉验证", type="primary"):
            with st.spinner("正在执行 60 次模型训练与验证，预计需要约 30 秒..."):
                comparison, fold_scores = cached_validation_crosscheck(data)
            st.session_state["validation_crosscheck"] = (comparison, fold_scores)
        if "validation_crosscheck" in st.session_state:
            comparison, fold_scores = st.session_state["validation_crosscheck"]
            model_names = {
                "dummy_prior": "先验概率基线",
                "logistic_regression": "类别加权逻辑回归",
                "pca_logistic": "PCA + 逻辑回归",
                "random_forest": "类别加权随机森林",
            }
            strategy_names = {"repeated_stratified": "重复分层", "temporal_expanding": "时间扩展窗口"}
            comparison_view = comparison.assign(
                model=lambda frame: frame["model"].map(model_names), validation_strategy=lambda frame: frame["validation_strategy"].map(strategy_names)
            ).rename(
                columns={
                    "model": "模型",
                    "validation_strategy": "验证策略",
                    "pr_auc_mean": "PR-AUC 均值",
                    "pr_auc_std": "PR-AUC 标准差",
                    "pr_auc_ci95_half_width": "PR-AUC 95% 区间半宽",
                    "balanced_accuracy_mean": "平衡准确率均值",
                    "roc_auc_mean": "ROC-AUC 均值",
                    "recall_mean": "召回率均值",
                    "precision_mean": "精确率均值",
                    "fit_time_mean_seconds": "平均训练时间（秒）",
                    "validation_folds": "验证折数",
                }
            )
            st.plotly_chart(
                px.bar(
                    comparison_view,
                    x="PR-AUC 均值",
                    y="模型",
                    color="验证策略",
                    error_x="PR-AUC 95% 区间半宽",
                    barmode="group",
                    orientation="h",
                    title="IID 与时间外推 PR-AUC 交叉验证",
                ),
                width="stretch",
            )
            st.dataframe(comparison_view.round(4), width="stretch", hide_index=True)
            st.caption("95% 区间按逐折均值的正态近似计算，仅用于描述波动；重复折并非相互独立。时间外推验证仍不等同于外部晶圆厂验证。")
            st.download_button("下载交叉验证逐折审计结果", fold_scores.to_csv(index=False), "SECOM_双策略交叉验证结果.csv", "text/csv")
    st.stop()

uploaded = st.sidebar.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])
if uploaded is None:
    data = pd.read_csv(ROOT / "data" / "demo_process_data.csv")
    st.sidebar.info("当前展示可复现的合成演示数据")
else:
    try:
        data = pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"无法读取上传文件：{exc}")
        st.stop()
data = clean_data(data)
missing = validate_data(data)
if missing:
    st.error("缺少必要字段：" + ", ".join(missing))
    st.stop()

equipment = st.sidebar.multiselect("设备筛选", sorted(data["equipment"].unique()), default=sorted(data["equipment"].unique()))
filtered = data[data["equipment"].isin(equipment)].copy()
if filtered.empty:
    st.warning("请至少选择一台存在有效记录的设备。")
    st.stop()
lsl = st.sidebar.number_input("良率规格下限 LSL（%）", value=90.0, step=0.5)
usl = st.sidebar.number_input("良率规格上限 USL（%）", value=100.0, step=0.5)
if lsl >= usl:
    st.error("良率规格下限 LSL 必须小于规格上限 USL。")
    st.stop()

summary = capability(filtered["yield_pct"], lsl, usl)
cards = st.columns(5)
cards[0].metric("记录数", f"{len(filtered):,}")
cards[1].metric("平均良率", f"{summary['mean']:.2f}%")
cards[2].metric("标准差", f"{summary['std']:.2f}")
cards[3].metric("Cpk", f"{summary['cpk']:.2f}" if pd.notna(summary["cpk"]) else "N/A")
cards[4].metric("低于 LSL", f"{(filtered['yield_pct'] < lsl).sum():,}")

overview, spc_tab, capability_tab, anomaly_tab, drift_tab, model_tab = st.tabs(["总览", "SPC 控制图", "过程能力", "异常筛查", "漂移监控", "影响因素"])
with overview:
    left, right = st.columns([1.5, 1])
    with left:
        trend = filtered.reset_index(drop=True)
        fig = px.line(
            trend, y="yield_pct", color="equipment", title="良率变化趋势", labels={"yield_pct": "良率（%）", "equipment": "设备", "index": "样本顺序"}
        )
        fig.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
        st.plotly_chart(fig, width="stretch")
    with right:
        by_eq = filtered.groupby("equipment", as_index=False)["yield_pct"].agg(["mean", "std", "count"]).reset_index()
        st.subheader("设备对比")
        st.dataframe(
            by_eq.rename(columns={"equipment": "设备", "mean": "平均值", "std": "标准差", "count": "样本数"}).round(3), width="stretch", hide_index=True
        )
        st.plotly_chart(
            px.box(filtered, x="equipment", y="yield_pct", color="equipment", labels={"equipment": "设备", "yield_pct": "良率（%）"}), width="stretch"
        )
with spc_tab:
    metric = st.selectbox("选择 SPC 监控指标", [c for c in filtered.select_dtypes("number").columns if c not in {"defect_count"}], index=0)
    baseline_size = st.number_input("Phase I 基准样本数", min_value=2, max_value=len(filtered), value=min(25, len(filtered)), step=1)
    chart = spc_chart(filtered[metric], baseline_size=int(baseline_size))
    individuals = go.Figure()
    individuals.add_trace(go.Scatter(x=chart["sample"], y=chart["value"], mode="lines+markers", name=metric))
    for column in ["center", "ucl", "lcl"]:
        names = {"center": "中心线", "ucl": "个体值 UCL", "lcl": "个体值 LCL"}
        individuals.add_trace(go.Scatter(x=chart["sample"], y=chart[column], mode="lines", name=names[column], line_dash="dash"))
    individuals.add_vline(x=int(baseline_size) + 0.5, line_dash="dot", annotation_text="Phase II 开始")
    individuals.update_layout(title="Individuals 个体值控制图", xaxis_title="样本顺序", yaxis_title=metric)
    mr_figure = go.Figure()
    mr_figure.add_trace(go.Scatter(x=chart["sample"], y=chart["moving_range"], mode="lines+markers", name="移动极差"))
    for column in ["mr_center", "mr_ucl", "mr_lcl"]:
        names = {"mr_center": "MR 中心线", "mr_ucl": "MR UCL", "mr_lcl": "MR LCL"}
        mr_figure.add_trace(go.Scatter(x=chart["sample"], y=chart[column], mode="lines", name=names[column], line_dash="dash"))
    mr_figure.add_vline(x=int(baseline_size) + 0.5, line_dash="dot", annotation_text="Phase II 开始")
    mr_figure.update_layout(title="Moving Range 移动极差控制图", xaxis_title="样本顺序", yaxis_title="相邻样本移动极差")
    alerts = chart[chart["out_of_control"]]
    st.plotly_chart(individuals, width="stretch")
    st.plotly_chart(mr_figure, width="stretch")
    st.caption("I-MR 控制限由 Phase I 基准窗口估计并冻结：σ̂=MR̄/1.128，个体值控制限为均值±3σ̂，MR 上限为 3.267×MR̄。规则包括 WE-1 至 WE-4，以及移动极差超限。")
    st.warning(f"Phase II 检测到 {len(alerts)} 个特殊原因信号点") if len(alerts) else st.success("Phase II 未检测到特殊原因信号")
    st.dataframe(alerts, width="stretch", hide_index=True)
with capability_tab:
    st.subheader("分设备过程能力")
    rows = []
    for name, group in filtered.groupby("equipment"):
        rows.append({"equipment": name, **capability(group["yield_pct"], lsl, usl)})
    capability_view = pd.DataFrame(rows).rename(columns={"equipment": "设备", "count": "样本数", "mean": "平均值", "std": "标准差", "cp": "Cp", "cpk": "Cpk"})
    st.dataframe(capability_view.round(3), width="stretch", hide_index=True)
    st.info("Cp/Cpk 根据所选规格限和样本标准差计算，仅用于过程诊断，不应单独作为产品放行依据。")
with anomaly_tab:
    result = detect_anomalies(filtered)
    st.subheader("孤立森林异常筛查")
    st.dataframe(result[result["anomaly"]].head(30), width="stretch", hide_index=True)
    st.download_button("下载异常筛查结果", result.to_csv(index=False), "工艺异常筛查结果.csv", "text/csv")
with drift_tab:
    numeric_cols = [c for c in filtered.select_dtypes("number").columns if c not in {"defect_count"}]
    ordered = filtered.reset_index(drop=True)
    split = int(len(ordered) * 0.7)
    if split < 5 or len(ordered) - split < 5:
        st.info("漂移监控至少需要 10 条筛选后的有效记录。")
    else:
        drift = cached_drift_report(ordered.iloc[:split], ordered.iloc[split:], numeric_cols)
        st.subheader("基准批次与当前批次分布对比")
        st.caption("以前 70% 记录作为基准窗口、后 30% 作为当前窗口，使用 PSI 与 KS 检验筛查分布变化。")
        drift_view = drift.assign(状态=drift["status"].map({"drift": "明显漂移", "watch": "需要关注", "stable": "基本稳定"}))
        st.plotly_chart(
            px.bar(drift_view.head(20), x="psi", y="feature", color="状态", orientation="h", labels={"feature": "指标", "psi": "PSI"}), width="stretch"
        )
        st.dataframe(drift.round(5), width="stretch", hide_index=True)
        st.download_button("下载漂移监控报告", drift.to_csv(index=False), "工艺漂移监控报告.csv", "text/csv")
with model_tab:
    importance = feature_importance(filtered)
    st.subheader("良率影响因素估计")
    if importance.empty:
        st.info("影响因素分析至少需要 20 条字段完整的有效记录。")
    else:
        st.plotly_chart(
            px.bar(
                importance.head(12),
                x="importance",
                y="feature",
                orientation="h",
                labels={"importance": "重要性", "feature": "变量"},
                title="随机森林变量重要性",
            ),
            width="stretch",
        )
        st.dataframe(importance.rename(columns={"feature": "变量", "importance": "重要性"}).round(4), width="stretch", hide_index=True)
