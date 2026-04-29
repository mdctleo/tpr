#!/bin/bash
# ******************************************************************************
# run_pikachu_experiments.sh
#
# Run all PIKACHU experiments on CIC-IDS 2017:
#   1. Generate reduced dataset (cic_20_red.csv)
#   2. Compute KDE features
#   3. Run baseline PIKACHU
#   4. Run KDE-enhanced PIKACHU  
#   5. Run reduced dataset PIKACHU
#   6. Run reduced + KDE PIKACHU
#
# Usage:
#   chmod +x run_pikachu_experiments.sh
#   ./run_pikachu_experiments.sh
# ******************************************************************************

set -e  # Exit on error

# Configuration
DATASET="cic_20"
INPUT_DIR="dataset/cic"
INPUT_CSV="${INPUT_DIR}/cic_20.csv"
INPUT_CSV_RED="${INPUT_DIR}/cic_20_red.csv"
KDE_FILE="weights/kde_edge_features.pickle"

# PIKACHU hyperparameters
DIM=100          # Embedding dimension
TRAINWIN=25      # Training window (snapshots 0-24 = Monday)
EPOCHS=50        # GRU epochs
ITER=10          # Gradient descent iterations
LR=0.001         # Learning rate
SUPPORT=15       # Support set size

echo "============================================================"
echo "PIKACHU Experiments on CIC-IDS 2017"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Generate reduced dataset
# -----------------------------------------------------------------------------
echo "[Step 1/6] Generating reduced dataset (one edge per src,dst pair)..."
if [ ! -f "$INPUT_CSV_RED" ]; then
    python generate_reduced_csv.py \
        --input "$INPUT_CSV" \
        --output "$INPUT_CSV_RED"
else
    echo "  Reduced CSV already exists: $INPUT_CSV_RED"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 2: Compute KDE features
# -----------------------------------------------------------------------------
echo "[Step 2/6] Computing KDE (DPGMM) edge features..."
if [ ! -f "$KDE_FILE" ]; then
    python compute_kde_features.py \
        --input "$INPUT_CSV" \
        --trainwin $TRAINWIN \
        --kde_dim 20 \
        --n_components 10 \
        --weight_concentration_prior 0.1 \
        --output "$KDE_FILE"
else
    echo "  KDE features already exist: $KDE_FILE"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 3: Run baseline PIKACHU
# -----------------------------------------------------------------------------
echo "[Step 3/6] Running baseline PIKACHU..."
python main.py \
    -ip "$INPUT_DIR/" \
    -d "$DATASET" \
    -k $DIM \
    -w $TRAINWIN \
    -e $EPOCHS \
    -i $ITER \
    -r $LR \
    -s $SUPPORT \
    -t True
echo ""

# -----------------------------------------------------------------------------
# Step 4: Run KDE-enhanced PIKACHU
# -----------------------------------------------------------------------------
echo "[Step 4/6] Running KDE-enhanced PIKACHU..."
python main.py \
    -ip "$INPUT_DIR/" \
    -d "$DATASET" \
    -k $DIM \
    -w $TRAINWIN \
    -e $EPOCHS \
    -i $ITER \
    -r $LR \
    -s $SUPPORT \
    -t False \
    --kde \
    --kde_file "$KDE_FILE" \
    --kde_dim 20
echo ""

# -----------------------------------------------------------------------------
# Step 5: Run reduced dataset PIKACHU
# -----------------------------------------------------------------------------
echo "[Step 5/6] Running PIKACHU on reduced dataset..."
python main.py \
    -ip "$INPUT_DIR/" \
    -d "$DATASET" \
    -k $DIM \
    -w $TRAINWIN \
    -e $EPOCHS \
    -i $ITER \
    -r $LR \
    -s $SUPPORT \
    -t True \
    --red
echo ""

# -----------------------------------------------------------------------------
# Step 6: Run reduced + KDE PIKACHU
# -----------------------------------------------------------------------------
echo "[Step 6/6] Running KDE-enhanced PIKACHU on reduced dataset..."
python main.py \
    -ip "$INPUT_DIR/" \
    -d "$DATASET" \
    -k $DIM \
    -w $TRAINWIN \
    -e $EPOCHS \
    -i $ITER \
    -r $LR \
    -s $SUPPORT \
    -t False \
    --red \
    --kde \
    --kde_file "$KDE_FILE" \
    --kde_dim 20
echo ""

echo "============================================================"
echo "All experiments complete!"
echo "Results saved in: results/results.txt"
echo "============================================================"
