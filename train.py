"""使用实验数据训练 6PPD/6PPD-Q 降解预测模型。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge


BASE_DIR = Path(__file__).resolve().parent
CONFIG = {
    "data_file": "degradation_data.csv",
    "feature_cols": [
        "时间_d",
        "介质",
        "氧化还原条件",
        "温度_C",
        "pH",
        "土壤含水率",
        "Eh_mV",
        "土壤foc",
        "DO_mg_L",
        "光照因子",
        "NOM_mg_L",
        "硝酸盐_mmol_L",
    ],
    "target_cols": ["6PPD_mg_kg", "6PPD_Q_mg_kg"],
    "group_col": "试验批次",
    "time_col": "时间_d",
    "test_size": 0.20,
    "random_seed": 42,
    "cv_folds": 5,
    "model_dir": "trained_model",
}


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _make_estimator(name: str, numeric_columns: list[str], categorical_columns: list[str], seed: int) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, numeric_columns), ("categorical", categorical_pipeline, categorical_columns)],
        remainder="drop",
    )
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
    }
    regressor = MultiOutputRegressor(models[name])
    target_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline([("preprocess", preprocessor), ("model", target_regressor)])


def _safe_r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred, multioutput="uniform_average"))


def _select_columns(data: pd.DataFrame, config: dict) -> tuple[list[str], list[str], list[str]]:
    targets = list(config["target_cols"])
    missing_targets = [column for column in targets if column not in data.columns]
    if missing_targets:
        raise KeyError(f"缺少目标列: {missing_targets}")
    requested = config.get("feature_cols")
    excluded = set(targets) | {config.get("group_col"), config.get("time_col")}
    features = list(requested) if requested else [column for column in data.columns if column not in excluded]
    if not features:
        raise ValueError("没有可用特征列。请设置 feature_cols 或在 CSV 中添加环境变量列。")
    missing_features = [column for column in features if column not in data.columns]
    if missing_features:
        raise KeyError(f"缺少特征列: {missing_features}")
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(data[column])]
    categorical = [column for column in features if column not in numeric]
    return features, numeric, categorical


def _split_data(data: pd.DataFrame, features: list[str], targets: list[str], config: dict):
    x_data = data[features]
    y_data = data[targets]
    group_column = config.get("group_col")
    if group_column and group_column in data.columns and data[group_column].nunique() >= 2:
        groups = data[group_column].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=config["test_size"], random_state=config["random_seed"])
        train_index, test_index = next(splitter.split(x_data, y_data, groups))
        return x_data.iloc[train_index], x_data.iloc[test_index], y_data.iloc[train_index], y_data.iloc[test_index], groups.iloc[train_index]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=config["test_size"], random_state=config["random_seed"]
    )
    return x_train, x_test, y_train, y_test, None


def train_dataframe(data: pd.DataFrame, config: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """训练、验证并返回可持久化 bundle 和模型对比表。"""
    settings = {**CONFIG, **(config or {})}
    features, numeric, categorical = _select_columns(data, settings)
    required = features + settings["target_cols"]
    clean = data.dropna(subset=settings["target_cols"]).copy()
    if len(clean) < 15:
        raise ValueError("至少需要 15 行含目标值的数据；建议每个试验批次至少 5 个时间点。")
    if (clean[settings["target_cols"]] < 0).any().any():
        raise ValueError("目标浓度不能为负；请先检查检出限替代值或数据单位。")

    x_train, x_test, y_train, y_test, train_groups = _split_data(clean, features, settings["target_cols"], settings)
    max_folds = min(settings["cv_folds"], len(x_train))
    if train_groups is not None:
        max_folds = min(max_folds, train_groups.nunique())
        cv = GroupKFold(n_splits=max_folds)
    else:
        cv = KFold(n_splits=max_folds, shuffle=True, random_state=settings["random_seed"])
    if max_folds < 3:
        raise ValueError("训练集至少需要 3 个独立批次，或增加样本量后再训练。")

    rows = []
    candidates = {}
    for name in ("Ridge", "RandomForest", "ExtraTrees"):
        estimator = _make_estimator(name, numeric, categorical, settings["random_seed"])
        cv_result = cross_validate(
            estimator,
            x_train,
            y_train,
            cv=cv,
            groups=train_groups,
            scoring="r2",
            n_jobs=1,
            error_score="raise",
        )
        estimator.fit(x_train, y_train)
        prediction = np.maximum(estimator.predict(x_test), 0.0)
        rows.append({
            "模型": name,
            "测试集_MAE": mean_absolute_error(y_test, prediction),
            "测试集_RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "测试集_R2": _safe_r2(y_test, prediction),
            "CV_R2_均值": float(np.mean(cv_result["test_score"])),
            "CV_R2_标准差": float(np.std(cv_result["test_score"])),
        })
        candidates[name] = estimator

    comparison = pd.DataFrame(rows).sort_values("CV_R2_均值", ascending=False).reset_index(drop=True)
    best_name = comparison.loc[0, "模型"]
    final_model = clone(candidates[best_name]).fit(clean[features], clean[settings["target_cols"]])
    bundle = {
        "bundle_version": 2,
        "model": final_model,
        "feature_cols": features,
        "target_cols": list(settings["target_cols"]),
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "metrics": comparison.iloc[0].to_dict(),
        "training_rows": len(clean),
        "group_col": settings.get("group_col"),
    }
    return bundle, comparison


def main() -> None:
    settings = CONFIG.copy()
    data_path = _resolve_path(settings["data_file"])
    if not data_path.exists():
        raise FileNotFoundError(f"未找到 {data_path}。请复制 degradation_data_template.csv 并在 CONFIG 中设置 data_file。")
    data = pd.read_csv(data_path)
    bundle, comparison = train_dataframe(data, settings)
    print("\n模型对比：")
    print(comparison.round(4).to_string(index=False))
    print("\n注意：CV R² <= 0 表示现有数据不足以支持该预测任务，应增加独立批次而非仅调参。")
    output_dir = _resolve_path(settings["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "degradation_model.pkl"
    with model_path.open("wb") as file_handle:
        pickle.dump(bundle, file_handle)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n最佳模型：{bundle['metrics']['模型']}；已保存至 {model_path}")


if __name__ == "__main__":
    main()
