#!/bin/bash
# CANDLE 訓練啟動腳本 for CL3AN_id20

set -e

PROJECT_DIR="/raid/ron/ALN_768/candle-main"
DATASET_ROOT="/raid/ron/ALN_768/dataset/CL3AN_id20"
CHECKPOINT_DIR="$PROJECT_DIR/train_ckpt_cl3an_id20"

echo "=========================================="
echo "CANDLE 訓練配置檢查"
echo "=========================================="

# 檢查資料集
echo "✓ 檢查訓練資料集..."
TRAIN_INPUT_COUNT=$(find "$DATASET_ROOT/train/input" -name "*.png" 2>/dev/null | wc -l)
TRAIN_NORMAL_COUNT=$(find "$DATASET_ROOT/train/normals" -name "*.npy" 2>/dev/null | wc -l)
TRAIN_GT_COUNT=$(find "$DATASET_ROOT/train/gt" -name "*.png" 2>/dev/null | wc -l)

echo "  訓練輸入: $TRAIN_INPUT_COUNT"
echo "  訓練法向量: $TRAIN_NORMAL_COUNT"
echo "  訓練真值: $TRAIN_GT_COUNT"

if [ $TRAIN_INPUT_COUNT -eq 0 ] || [ $TRAIN_NORMAL_COUNT -eq 0 ] || [ $TRAIN_GT_COUNT -eq 0 ]; then
    echo "❌ 訓練資料集不完整!"
    exit 1
fi

echo ""
echo "✓ 檢查測試資料集..."
TEST_INPUT_COUNT=$(find "$DATASET_ROOT/test/input" -name "*.png" 2>/dev/null | wc -l)
TEST_NORMAL_COUNT=$(find "$DATASET_ROOT/test/normals" -name "*.npy" 2>/dev/null | wc -l)
TEST_GT_COUNT=$(find "$DATASET_ROOT/test/gt" -name "*.png" 2>/dev/null | wc -l)

echo "  測試輸入: $TEST_INPUT_COUNT"
echo "  測試法向量: $TEST_NORMAL_COUNT"
echo "  測試真值: $TEST_GT_COUNT"

if [ $TEST_INPUT_COUNT -eq 0 ] || [ $TEST_NORMAL_COUNT -eq 0 ] || [ $TEST_GT_COUNT -eq 0 ]; then
    echo "❌ 測試資料集不完整!"
    exit 1
fi

echo ""
echo "✓ 檢查模型檔案..."
if [ ! -f "$PROJECT_DIR/model.py" ]; then
    echo "❌ model.py 不存在"
    exit 1
fi

echo "✓ 檢查訓練腳本..."
if [ ! -f "$PROJECT_DIR/train.py" ]; then
    echo "❌ train.py 不存在"
    exit 1
fi

echo ""
echo "=========================================="
echo "資料統計"
echo "=========================================="
echo "訓練: $TRAIN_INPUT_COUNT 圖 + $TRAIN_NORMAL_COUNT 法向量 + $TRAIN_GT_COUNT 真值"
echo "測試: $TEST_INPUT_COUNT 圖 + $TEST_NORMAL_COUNT 法向量 + $TEST_GT_COUNT 真值"
echo ""

# 建立檢查點目錄
mkdir -p "$CHECKPOINT_DIR"
echo "✓ 檢查點目錄: $CHECKPOINT_DIR"

echo ""
echo "=========================================="
echo "準備啟動訓練"
echo "=========================================="
echo ""
echo "命令: cd $PROJECT_DIR && python train.py"
echo ""
echo "推薦參數 (可在 options.py 修改):"
echo "  - batch_size: 1 (默認)"
echo "  - epochs: 300"
echo "  - patch_size: 384"
echo "  - lr: 2e-4"
echo ""
read -p "按 Enter 開始訓練..." confirm

cd "$PROJECT_DIR"
python train.py
