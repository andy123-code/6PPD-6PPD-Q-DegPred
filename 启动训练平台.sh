#!/bin/bash
# 6PPD 土壤释放预测 —— 训练平台启动脚本
# 双击运行或在终端执行: bash 启动训练平台.sh

cd "$(dirname "$0")"

echo "========================================"
echo "  6PPD 土壤释放预测 —— 训练平台"
echo "========================================"
echo ""
echo "  浏览器打开: http://localhost:8501"
echo "  停止: 关闭此窗口 或 Ctrl+C"
echo ""

streamlit run app_v2.py --server.port 8501
