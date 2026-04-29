#!/bin/bash
# Analyze KDE results (temp_kde/) without touching baseline logs (log/).
# Temporarily swaps ../temp to point at temp_kde via symlink.
set -e

BASEDIR="/scratch/asawan15/HyperVision"
cd "$BASEDIR/result_analyze"

# Backup original temp if it's a real directory (not a symlink)
if [ -d "$BASEDIR/temp" ] && [ ! -L "$BASEDIR/temp" ]; then
    mv "$BASEDIR/temp" "$BASEDIR/temp_baseline_backup"
fi

# Symlink temp -> temp_kde (absolute paths)
ln -sfn "$BASEDIR/temp_kde" "$BASEDIR/temp"

# Create kde log/figure dirs
mkdir -p log_kde figure_kde

# Run analysis for all 3 groups, saving to log_kde/
for g in brute lrscan misc; do
    echo "=== Analyzing group: $g (KDE) ==="
    mkdir -p log_kde/$g
    python batch_analyzer.py -g $g 2>&1 || true
    # Move results from log/{group} to log_kde/{group}
    if [ -d log/$g ]; then
        cp -f log/$g/*.log log_kde/$g/ 2>/dev/null || true
    fi
done

# Restore original temp
rm -f "$BASEDIR/temp"
if [ -d "$BASEDIR/temp_baseline_backup" ]; then
    mv "$BASEDIR/temp_baseline_backup" "$BASEDIR/temp"
fi

echo ""
echo "=== KDE analysis complete. Logs in log_kde/ ==="
echo ""

# Summarize with comparison
python summarize_results.py --log-dir log --groups brute lrscan misc \
    --compare-dir log_kde --compare-label "KDE-Enhanced"
