"""使用 degradation_model.pkl 对新实验条件进行批量预测。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model" / "degradation_model.pkl"


def load_model(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型：{model_path}。请先运行 python3 train.py。")
    with model_path.open("rb") as file_handle:
        bundle = pickle.load(file_handle)
    if bundle.get("bundle_version") != 2:
        raise ValueError("该模型是旧版格式。请使用新的 train.py 重新训练。")
    return bundle


def predict_dataframe(data: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    missing = [column for column in bundle["feature_cols"] if column not in data.columns]
    if missing:
        raise KeyError(f"预测数据缺少特征列: {missing}")
    prediction = np.maximum(bundle["model"].predict(data[bundle["feature_cols"]]), 0.0)
    result = data.copy()
    for index, target in enumerate(bundle["target_cols"]):
        result[f"预测_{target}"] = prediction[:, index]
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python3 predict.py <新数据.csv> [--save]")
        raise SystemExit(1)
    input_path = Path(sys.argv[1]).resolve()
    bundle = load_model()
    result = predict_dataframe(pd.read_csv(input_path), bundle)
    print(result[[f"预测_{target}" for target in bundle["target_cols"]]].round(6).to_string(index=False))
    if "--save" in sys.argv:
        output_path = input_path.with_name(f"{input_path.stem}_predicted.csv")
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存：{output_path}")


if __name__ == "__main__":
    main()
