"""
6PPD 土壤释放预测 — 可视化训练平台
===================================
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="6PPD 释放预测训练器", layout="wide")

# ---- 中文支撑 ----
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- session state ----
for key in ["df", "trained_bundle", "predict_df"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# 工具函数
# ============================================================

def build_model(name, n_estimators, max_depth, learning_rate, alpha, n_neighbors):
    """根据用户选择的超参数构建模型"""
    models = {
        "随机森林": RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=3, random_state=42, n_jobs=-1),
        "梯度提升": GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, random_state=42),
        "Ridge": Ridge(alpha=alpha),
        "Lasso": Lasso(alpha=alpha, max_iter=5000),
        "SVR(RBF)": SVR(kernel="rbf", C=alpha, gamma="scale"),
        "SVR(Linear)": SVR(kernel="linear", C=alpha),
        "KNN": KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance"),
    }
    return models[name]


NEEDS_SCALE = {"SVR(RBF)", "SVR(Linear)", "Ridge", "Lasso", "KNN"}


def train_and_eval(model, X_train, y_train, X_test, y_test, name, scaler):
    needs = name in NEEDS_SCALE
    if needs:
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_te = X_train.values, X_test.values

    if y_train.shape[1] > 1:
        model = MultiOutputRegressor(model)

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    y_true = y_test.values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring="r2")
    return model, mae, rmse, r2, cv.mean(), cv.std(), y_pred, y_true, needs


# ============================================================
# 页面
# ============================================================

st.title("6PPD 土壤介质释放预测 —— 可视化训练平台")
st.caption("上传你的数据 → 选特征 → 训练模型 → 对比效果 → 预测新数据")

tabs = st.tabs(["1. 数据加载", "2. 模型训练", "3. 结果分析", "4. 预测新数据"])

# ==================== TAB 1: 数据加载 ====================
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上传数据")
        uploaded = st.file_uploader("上传 CSV 文件", type=["csv"], key="upload_main")

        if uploaded:
            st.session_state.df = pd.read_csv(uploaded)
            st.success(f"已加载 {len(st.session_state.df)} 行, {len(st.session_state.df.columns)} 列")
        else:
            st.info("或使用内置示例数据")
            if st.button("加载示例数据"):
                from pathlib import Path
                sample_path = Path(__file__).parent / "release_data.csv"
                if sample_path.exists():
                    st.session_state.df = pd.read_csv(sample_path)
                else:
                    # 用物理模型生成
                    from ppd_release_model import TWPReleaseModel, SoilTransportModel
                    rows = []
                    rng = np.random.default_rng(42)
                    for _ in range(200):
                        Tc = rng.uniform(5, 35)
                        rain = rng.exponential(4)
                        moist = np.clip(rng.normal(0.30, 0.12), 0.05, 0.50)
                        foc = np.clip(rng.lognormal(-3.5, 0.6), 0.003, 0.10)
                        ph = rng.uniform(5.0, 8.5)
                        size = rng.choice([25, 50, 75, 100, 150])
                        pct = rng.uniform(0.5, 2.0)
                        days = rng.uniform(1, 365)
                        twp = TWPReleaseModel(particle_radius_um=size / 2, loading_wt_pct=pct, twp_mass_g=1.0)
                        m = twp.cumulative_release(days, Tc, moist)
                        soil = SoilTransportModel({"foc": foc, "water_content": moist})
                        _, cs = soil.steady_state(m / max(days, 1), rain, Tc)
                        rows.append([Tc, rain, moist, foc, ph, size, pct, days, m, cs])
                    col_names = ["温度_C", "降水_mm_d", "土壤含水率", "土壤foc", "土壤pH",
                                 "TWP粒径_um", "6PPD含量_pct", "老化天数", "释放量_mg", "土壤浓度_mg_kg"]
                    st.session_state.df = pd.DataFrame(rows, columns=col_names)
                st.success(f"已加载 {len(st.session_state.df)} 行示例数据")

    with col2:
        if st.session_state.df is not None:
            st.subheader("数据预览")
            st.dataframe(st.session_state.df.head(15), use_container_width=True)
            st.caption(f"共 {len(st.session_state.df)} 行 × {len(st.session_state.df.columns)} 列")

            st.subheader("统计摘要")
            st.dataframe(st.session_state.df.describe().round(3), use_container_width=True)


# ==================== TAB 2: 模型训练 ====================
with tabs[1]:
    if st.session_state.df is None:
        st.warning("请先在 Tab 1 加载数据")
    else:
        df = st.session_state.df
        all_cols = list(df.columns)

        col_a, col_b = st.columns(2)
        with col_a:
            feature_cols = st.multiselect(
                "选择特征列 (X)", all_cols,
                default=[c for c in all_cols if c not in ["释放量_mg", "土壤浓度_mg_kg"]]
            )
        with col_b:
            target_cols = st.multiselect(
                "选择目标列 (y)", all_cols,
                default=[c for c in ["释放量_mg", "土壤浓度_mg_kg"] if c in all_cols]
            )

        cat_cols = st.multiselect(
            "类别列（文字列需LabelEncode）", feature_cols, default=[]
        )

        st.divider()

        st.subheader("选择要对比的模型 + 超参数")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_estimators = st.slider("树的数量 (RF/GB)", 50, 500, 200, 50)
            max_depth = st.slider("最大深度 (RF/GB)", 3, 20, 8)
        with c2:
            learning_rate = st.slider("学习率 (GB)", 0.01, 0.30, 0.05, 0.01)
            n_neighbors = st.slider("KNN邻居数", 2, 20, 5)
        with c3:
            alpha = st.slider("alpha / C (Ridge/Lasso/SVR)", 0.001, 100.0, 1.0, 0.1,
                              format="%.3f")
        with c4:
            test_size = st.slider("测试集比例", 0.10, 0.40, 0.20, 0.05)
            active_models = st.multiselect(
                "模型", ["随机森林", "梯度提升", "Ridge", "Lasso", "SVR(RBF)", "SVR(Linear)", "KNN"],
                default=["随机森林", "梯度提升", "Ridge"]
            )

        if st.button("开始训练", type="primary", use_container_width=True):
            if not feature_cols or not target_cols:
                st.error("请至少选一个特征列和一个目标列")
            else:
                # 数据预处理
                X = df[feature_cols].copy()
                y = df[target_cols].copy()

                encoders = {}
                for col in cat_cols:
                    if col in X.columns:
                        le = LabelEncoder()
                        X[col] = X[col].astype(str)
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                st.info(f"训练集 {len(X_train)} 行 | 测试集 {len(X_test)} 行 | "
                       f"特征 {len(feature_cols)} | 目标 {len(target_cols)}")

                # 训练所有选中模型
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, name in enumerate(active_models):
                    status.text(f"训练中: {name} ...")
                    m = build_model(name, n_estimators, max_depth, learning_rate, alpha, n_neighbors)
                    model, mae, rmse, r2, cv_r2, cv_std, yp, yt, needs = train_and_eval(
                        m, X_train, y_train, X_test, y_test, name, StandardScaler()
                    )
                    results.append({
                        "模型": name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R²": r2,
                        "CV R²": cv_r2,
                        "CV std": cv_std,
                        "需要标准化": needs,
                        "trained_model": model,
                        "y_pred": yp,
                        "y_true": yt,
                        "feature_cols": feature_cols,
                        "target_cols": target_cols,
                        "encoders": encoders,
                    })
                    progress.progress((i + 1) / len(active_models))

                progress.empty()
                status.text("训练完成！")

                # 选最优，全量重训
                best = max(results, key=lambda r: r["CV R²"])
                best_base = build_model(best["模型"], n_estimators, max_depth, learning_rate, alpha, n_neighbors)
                if best["需要标准化"]:
                    X_all = StandardScaler().fit_transform(X.values)
                    scaler_final = StandardScaler()
                    scaler_final.fit(X.values)
                else:
                    X_all = X.values
                    scaler_final = None
                if y.shape[1] > 1:
                    best_base = MultiOutputRegressor(best_base)
                best_base.fit(X_all, y.values)

                st.session_state.trained_bundle = {
                    "model": best_base,
                    "scaler": scaler_final,
                    "needs_scale": best["需要标准化"],
                    "feature_cols": feature_cols,
                    "target_cols": target_cols,
                    "encoders": encoders,
                    "is_multi": y.shape[1] > 1,
                    "best_name": best["模型"],
                    "cv_r2": best["CV R²"],
                    "results": results,
                }
                st.success(f"最佳模型: {best['模型']} (CV R² = {best['CV R²']:.4f})，已保存")

                # 下载按钮
                buf = io.BytesIO()
                export = {k: v for k, v in st.session_state.trained_bundle.items()
                          if k != "results"}
                pickle.dump(export, buf)
                st.download_button("下载模型文件 (model.pkl)", buf.getvalue(),
                                   "model.pkl", "application/octet-stream")


# ==================== TAB 3: 结果分析 ====================
with tabs[2]:
    bundle = st.session_state.trained_bundle
    if bundle is None or "results" not in bundle:
        st.warning("请先在 Tab 2 完成训练")
    else:
        results = bundle["results"]
        df_r = pd.DataFrame([{k: v for k, v in r.items()
                              if k not in ["trained_model", "y_pred", "y_true",
                                           "feature_cols", "target_cols", "encoders"]}
                             for r in results])
        df_r = df_r.sort_values("CV R²", ascending=False)

        st.subheader("模型对比")

        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.dataframe(df_r.round(4), use_container_width=True, hide_index=True)

        with col_b:
            fig, ax = plt.subplots(figsize=(8, 5))
            # 排序后用 horizontal bar
            names = [r["模型"] for r in results]
            cv_vals = [r["CV R²"] for r in results]
            test_r2 = [r["R²"] for r in results]

            x = np.arange(len(names))
            w = 0.35
            bars1 = ax.barh(x + w/2, cv_vals, w, label="CV R² (5折)", color="#4C72B0")
            bars2 = ax.barh(x - w/2, test_r2, w, label="测试集 R²", color="#DD8452")
            ax.set_yticks(x)
            ax.set_yticklabels(names)
            ax.set_xlabel("R²")
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.legend(loc="lower right")
            ax.set_title("模型 R² 对比")
            st.pyplot(fig)

        st.divider()

        # 选一个模型看详情
        model_names = [r["模型"] for r in results]
        selected = st.selectbox("选择一个模型查看详情", model_names,
                                 index=model_names.index(bundle["best_name"]))

        sel_res = next(r for r in results if r["模型"] == selected)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("预测 vs 真实")

            yp = sel_res["y_pred"]
            yt = sel_res["y_true"]
            targets = sel_res["target_cols"]
            n_targets = yt.shape[1] if len(yt.shape) > 1 else 1

            if n_targets > 1 and len(yt.shape) > 1:
                # 多目标，每个画一个子图
                fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5))
                if n_targets == 1:
                    axes = [axes]
                for i, ax in enumerate(axes):
                    ax.scatter(yt[:, i], yp[:, i], alpha=0.6, s=20, edgecolors="none")
                    lims = [min(yt[:, i].min(), yp[:, i].min()),
                            max(yt[:, i].max(), yp[:, i].max())]
                    ax.plot(lims, lims, "r--", linewidth=1)
                    ax.set_xlabel(f"真实 {targets[i]}")
                    ax.set_ylabel(f"预测 {targets[i]}")
                    ax.set_title(f"{targets[i]}")
            else:
                yp_flat = yp.ravel()
                yt_flat = yt.ravel() if len(yt.shape) > 1 else yt
                fig, ax = plt.subplots(figsize=(5, 4.5))
                ax.scatter(yt_flat, yp_flat, alpha=0.6, s=20, edgecolors="none")
                lims = [min(yt_flat.min(), yp_flat.min()),
                        max(yt_flat.max(), yp_flat.max())]
                ax.plot(lims, lims, "r--", linewidth=1)
                ax.set_xlabel("真实值")
                ax.set_ylabel("预测值")
                ax.set_title(f"真实 vs 预测 ({targets[0]})")

            st.pyplot(fig)

        with col2:
            st.subheader("残差分布")
            if n_targets > 1 and len(yt.shape) > 1:
                fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5))
                if n_targets == 1:
                    axes = [axes]
                for i, ax in enumerate(axes):
                    r = yt[:, i] - yp[:, i]
                    ax.hist(r, bins=30, edgecolor="white", color="#4C72B0", alpha=0.8)
                    ax.axvline(0, color="red", linewidth=1, linestyle="--")
                    ax.set_xlabel("残差")
                    ax.set_ylabel("频数")
                    ax.set_title(f"残差分布 ({targets[i]})")
            else:
                yp_flat = yp.ravel()
                yt_flat = yt.ravel() if len(yt.shape) > 1 else yt
                residuals = yt_flat - yp_flat
                fig, ax = plt.subplots(figsize=(5, 4.5))
                ax.hist(residuals, bins=30, edgecolor="white", color="#4C72B0", alpha=0.8)
                ax.axvline(0, color="red", linewidth=1, linestyle="--")
                ax.set_xlabel("残差")
                ax.set_ylabel("频数")
                ax.set_title("残差分布")
            st.pyplot(fig)

        st.divider()

        st.subheader("特征重要性（排列重要性）")
        # 用随机森林评估重要性
        if "feature_cols" in sel_res and "y_true" in sel_res:
            import_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            needs = sel_res.get("需要标准化", False)
            from sklearn.model_selection import train_test_split as tts
            df_train = st.session_state.df[sel_res["feature_cols"]].copy()
            for col in sel_res.get("encoders", {}):
                if col in df_train.columns:
                    le = sel_res["encoders"][col]
                    df_train[col] = df_train[col].astype(str)
                    known = set(le.classes_)
                    df_train[col] = df_train[col].apply(lambda x: x if x in known else "unknown")
                    df_train[col] = le.transform(df_train[col])
            y_train = st.session_state.df[sel_res["target_cols"]].values[:, 0]  # 第一个目标

            if needs:
                sc = StandardScaler()
                X_imp = sc.fit_transform(df_train.values)
            else:
                X_imp = df_train.values

            import_model.fit(X_imp, y_train)
            perm = permutation_importance(import_model, X_imp, y_train,
                                          n_repeats=10, random_state=42, scoring="r2")

            importances = perm.importances_mean
            idx = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh([sel_res["feature_cols"][i] for i in idx][::-1],
                    importances[idx][::-1],
                    xerr=perm.importances_std[idx][::-1],
                    color="#55A868", edgecolor="none")
            ax.set_xlabel("重要性 (R² 下降)")
            ax.set_title("排列重要性")
            st.pyplot(fig)


# ==================== TAB 4: 预测新数据 ====================
with tabs[3]:
    bundle = st.session_state.trained_bundle
    if bundle is None:
        st.warning("请先在 Tab 2 完成训练，或上传已保存的模型")
        uploaded_model = st.file_uploader("上传 model.pkl", type=["pkl"])
        if uploaded_model:
            bundle = pickle.load(uploaded_model)
            st.session_state.trained_bundle = bundle
            st.success("模型已加载")

    if bundle is not None:
        st.subheader("输入新数据")

        mode = st.radio("输入方式", ["手动输入", "上传 CSV"], horizontal=True)

        if mode == "上传 CSV":
            pred_csv = st.file_uploader("上传待预测 CSV", type=["csv"], key="pred_csv_upload")
            if pred_csv:
                st.session_state.predict_df = pd.read_csv(pred_csv)
                st.dataframe(st.session_state.predict_df.head(10), use_container_width=True)
        else:
            st.write("逐行输入特征值：")
            manual_data = {}
            cols = st.columns(min(len(bundle["feature_cols"]), 4))
            for i, feat in enumerate(bundle["feature_cols"]):
                with cols[i % 4]:
                    manual_data[feat] = st.number_input(feat, value=22.0, format="%.3f")
            if st.button("用此行预测"):
                st.session_state.predict_df = pd.DataFrame([manual_data])

        if st.session_state.predict_df is not None:
            if st.button("执行预测", type="primary"):
                X_new = st.session_state.predict_df[bundle["feature_cols"]].copy()

                for col in bundle.get("encoders", {}):
                    if col in X_new.columns:
                        le = bundle["encoders"][col]
                        X_new[col] = X_new[col].astype(str)
                        known = set(le.classes_)
                        X_new[col] = X_new[col].apply(lambda x: x if x in known else "unknown")
                        X_new[col] = le.transform(X_new[col])

                if bundle.get("needs_scale") and bundle["scaler"] is not None:
                    X_arr = bundle["scaler"].transform(X_new.values)
                else:
                    X_arr = X_new.values

                y_pred = bundle["model"].predict(X_arr)

                result_df = st.session_state.predict_df.copy()
                if bundle["is_multi"]:
                    for i, t in enumerate(bundle["target_cols"]):
                        result_df[f"预测_{t}"] = y_pred[:, i].round(4)
                else:
                    result_df[f"预测_{bundle['target_cols'][0]}"] = y_pred.round(4)

                st.success("预测完成")
                st.dataframe(result_df, use_container_width=True)

                # 下载
                buf = io.BytesIO()
                result_df.to_csv(buf, index=False, encoding="utf-8-sig")
                st.download_button("下载预测结果 CSV", buf.getvalue(),
                                   "predictions.csv", "text/csv")
