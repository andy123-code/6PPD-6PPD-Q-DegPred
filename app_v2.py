"""6PPD/6PPD-Q 降解预测平台（新版 Streamlit 界面）。"""

import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from degradation_model import FateConditions, TwoSpeciesFateModel
from predict import predict_dataframe
from train import train_dataframe


st.set_page_config(page_title="6PPD 降解预测平台", layout="wide")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR / "saved_workspace"
TRAINING_DATA_PATH = WORKSPACE_DIR / "latest_training_data.csv"
MODEL_PATH = WORKSPACE_DIR / "latest_model.pkl"
COMPARISON_PATH = WORKSPACE_DIR / "latest_model_comparison.csv"
LOCAL_PERSISTENCE = st.context.headers.get("host", "").startswith(("localhost", "127.0.0.1"))

for key in ("training_data", "bundle", "prediction_result", "comparison"):
    st.session_state.setdefault(key, None)


def save_training_data(data: pd.DataFrame) -> None:
    if not LOCAL_PERSISTENCE:
        return
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(TRAINING_DATA_PATH, index=False, encoding="utf-8-sig")


def save_model_bundle(bundle: dict, comparison: pd.DataFrame) -> None:
    if not LOCAL_PERSISTENCE:
        return
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file_handle:
        pickle.dump(bundle, file_handle)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")


def load_saved_workspace() -> None:
    if not LOCAL_PERSISTENCE:
        return
    if st.session_state.training_data is None and TRAINING_DATA_PATH.exists():
        st.session_state.training_data = pd.read_csv(TRAINING_DATA_PATH)
    if st.session_state.bundle is None and MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as file_handle:
            st.session_state.bundle = pickle.load(file_handle)
    if st.session_state.comparison is None and COMPARISON_PATH.exists():
        st.session_state.comparison = pd.read_csv(COMPARISON_PATH)


def clear_saved_workspace() -> None:
    if LOCAL_PERSISTENCE:
        for path in (TRAINING_DATA_PATH, MODEL_PATH, COMPARISON_PATH):
            if path.exists():
                path.unlink()
    for key in ("training_data", "bundle", "prediction_result", "comparison"):
        st.session_state[key] = None


def download_csv(data: pd.DataFrame, name: str) -> None:
    buffer = io.BytesIO()
    data.to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button("下载 CSV", buffer.getvalue(), name, "text/csv")


load_saved_workspace()

st.title("6PPD / 6PPD-Q 土壤与水体降解预测")
st.caption("条件化机理模拟 + 分组交叉验证的实验数据模型。模型输出必须结合独立批次验证解释。")

with st.sidebar:
    if LOCAL_PERSISTENCE:
        st.header("本地记忆")
        if st.session_state.training_data is not None:
            st.write(f"训练数据：{len(st.session_state.training_data)} 行")
        else:
            st.write("训练数据：未保存")
        if st.session_state.bundle is not None:
            metrics = st.session_state.bundle.get("metrics", {})
            st.write(f"模型：{metrics.get('模型', '已保存')}")
            if "CV_R2_均值" in metrics:
                st.write(f"CV R²：{metrics['CV_R2_均值']:.4f}")
        else:
            st.write("模型：未保存")
        if st.button("清除本地记忆"):
            clear_saved_workspace()
            st.rerun()
    else:
        st.header("会话状态")
        st.write("公开版数据仅保存在当前浏览器会话中。")

tabs = st.tabs(["1. 数据与训练", "2. 批量预测", "3. 动态机理模拟"])

with tabs[0]:
    uploaded_data = st.file_uploader("上传实验 CSV", type="csv")
    if uploaded_data is not None:
        st.session_state.training_data = pd.read_csv(uploaded_data)
        save_training_data(st.session_state.training_data)
        if LOCAL_PERSISTENCE:
            st.success("训练数据已保存到本地，下次打开会自动恢复。")
    data = st.session_state.training_data
    if data is None:
        st.info("请使用 degradation_data_template.csv 作为字段模板。每一行应为一个培养瓶/地点在一个时间点的观测。")
    else:
        st.dataframe(data.head(20), use_container_width=True)
        columns = list(data.columns)
        target_columns = st.multiselect("目标列（浓度、半衰期或速率）", columns, default=[c for c in columns if "6PPD" in c])
        group_options = ["(无)"] + columns
        default_group = "试验批次" if "试验批次" in columns else "(无)"
        group_column = st.selectbox("独立批次列（推荐：培养瓶/采样点 ID）", group_options, index=group_options.index(default_group))
        excluded = set(target_columns) | ({group_column} if group_column != "(无)" else set())
        default_features = [c for c in columns if c not in excluded]
        feature_columns = st.multiselect("特征列", columns, default=default_features)
        if st.button("训练并比较模型", type="primary"):
            if not target_columns or not feature_columns:
                st.error("请至少选择一个目标列和一个特征列。")
            else:
                try:
                    bundle, comparison = train_dataframe(data, {
                        "target_cols": target_columns,
                        "feature_cols": feature_columns,
                        "group_col": None if group_column == "(无)" else group_column,
                    })
                    st.session_state.bundle = bundle
                    st.session_state.comparison = comparison
                    save_model_bundle(bundle, comparison)
                    st.success(f"最佳模型：{bundle['metrics']['模型']}；CV R² = {bundle['metrics']['CV_R2_均值']:.4f}")
                    if LOCAL_PERSISTENCE:
                        st.info("训练结果和模型已保存到本地，下次打开会自动恢复。")
                    st.dataframe(comparison.round(4), use_container_width=True)
                    model_buffer = io.BytesIO()
                    pickle.dump(bundle, model_buffer)
                    st.download_button("下载训练模型", model_buffer.getvalue(), "degradation_model.pkl", "application/octet-stream")
                except (KeyError, ValueError) as error:
                    st.error(str(error))
        elif st.session_state.comparison is not None:
            st.subheader("上次训练结果")
            st.dataframe(st.session_state.comparison.round(4), use_container_width=True)

with tabs[1]:
    bundle = st.session_state.bundle
    if bundle is None:
        st.info("请先在“数据与训练”页完成训练。")
    else:
        uploaded_prediction = st.file_uploader("上传待预测 CSV", type="csv", key="prediction_csv")
        if uploaded_prediction is not None:
            try:
                result = predict_dataframe(pd.read_csv(uploaded_prediction), bundle)
                st.session_state.prediction_result = result
                st.dataframe(result, use_container_width=True)
                download_csv(result, "degradation_predictions.csv")
            except KeyError as error:
                st.error(str(error))

with tabs[2]:
    first, second, third, fourth = st.columns(4)
    with first:
        medium = st.selectbox("介质", ["soil", "water"], format_func=lambda value: {"soil": "土壤", "water": "水体"}[value])
        redox_labels = {
            "aerobic": "好氧",
            "anaerobic": "厌氧",
            "nitrate_reducing": "硝酸盐还原",
            "sulfate_reducing": "硫酸盐还原",
            "iron_reducing": "铁还原",
            "sterilized": "灭菌",
        }
        redox = st.selectbox("氧化还原状态", list(redox_labels), format_func=redox_labels.get)
        temperature = st.number_input("温度 (°C)", value=22.0)
    with second:
        p_h = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
        moisture = st.number_input("土壤含水率", min_value=0.0, max_value=1.0, value=0.30)
        foc = st.number_input("有机碳分数 foc", min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
    with third:
        light = st.number_input("光照因子 (0-1)", min_value=0.0, max_value=1.0, value=0.0)
        ozone = st.number_input("臭氧/氧化因子 (0-1)", min_value=0.0, max_value=1.0, value=0.0)
        epfr = st.number_input("EPFR 因子 (0-1)", min_value=0.0, max_value=1.0, value=0.0)
        nom = st.number_input("NOM (mg/L)", min_value=0.0, value=0.0)
        nitrate = st.number_input("硝酸盐 (mmol/L)", min_value=0.0, value=0.0)
    with fourth:
        duration = st.number_input("模拟时长 (d)", min_value=0.1, value=60.0)
        initial_6ppd = st.number_input("初始 6PPD (mg)", min_value=0.0, value=5.0)
        initial_q = st.number_input("初始 6PPD-Q (mg)", min_value=0.0, value=0.0)
        input_6ppd = st.number_input("持续输入 6PPD (mg/d)", min_value=0.0, value=0.0)

    conditions = FateConditions(
        medium=medium, redox=redox, temperature_c=temperature, pH=p_h,
        moisture=moisture, organic_carbon_fraction=foc, light_factor=light,
        ozone_factor=ozone, epfr_factor=epfr, nom_mg_l=nom, nitrate_mmol_l=nitrate,
    )
    model = TwoSpeciesFateModel(conditions)
    simulation = model.simulate(duration, initial_6ppd, initial_q, input_6ppd_mg_d=input_6ppd)
    st.json(model.rates())
    st.dataframe(pd.DataFrame([model.mechanistic_features()]).T.rename(columns={0: "值"}), use_container_width=True)
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(simulation["time_d"], simulation["6PPD_total_mg"], label="6PPD")
    axis.plot(simulation["time_d"], simulation["6PPD_Q_total_mg"], label="6PPD-Q")
    axis.set(xlabel="时间 (d)", ylabel="总质量 (mg)")
    axis.legend()
    st.pyplot(fig)
    download_csv(simulation, "mechanistic_simulation.csv")
