#!/usr/bin/env bash
# ===========================================================================
#  KDE Dimension Grid Search
#  Generates KDE features at dims 5,10,15,25,30, creates configs, runs
#  HyperVision for all 43 attacks at each dim, and collects analysis logs.
#
#  Directory layout per dimension D:
#    kde_features_D/          — precomputed KDE CSVs (D columns)
#    configuration/{cat}_kde_D/ — JSON configs pointing to kde_features_D/
#    temp_kde_D/              — HyperVision result files
#    result_analyze/log_kde_D/ — analysis logs
#
#  Usage:
#    cd /scratch/asawan15/HyperVision/build
#    bash ../script/run_kde_grid_search.sh
#
#  To run only specific dims:
#    DIMS="5 10" bash ../script/run_kde_grid_search.sh
# ===========================================================================
set -euo pipefail

BASEDIR="/scratch/asawan15/HyperVision"
BUILDDIR="$BASEDIR/build"
DIMS="${DIMS:-5 10 15 25 30}"
ANALYZE_DIR="$BASEDIR/result_analyze"
BASELINE_LOG_DIR="$ANALYZE_DIR/log_baseline"

export LD_LIBRARY_PATH="$HOME/local/lib:$HOME/local/lib64:${LD_LIBRARY_PATH:-}"

# Ensure binary is built
cd "$BUILDDIR"
make -j$(nproc)

# Freeze baseline logs once and never compare against mutable log/
if [ ! -d "$BASELINE_LOG_DIR" ]; then
    if [ ! -d "$ANALYZE_DIR/log" ]; then
        echo "ERROR: Missing baseline logs at $ANALYZE_DIR/log"
        echo "Run baseline analysis first, then re-run this script."
        exit 1
    fi
    echo "[setup] Creating immutable baseline snapshot: $BASELINE_LOG_DIR"
    rm -rf "$BASELINE_LOG_DIR"
    cp -a "$ANALYZE_DIR/log" "$BASELINE_LOG_DIR"
fi

# ── Attack lists (same as run_all_43_kde.sh) ──
BRUTE="charrdos cldaprdos dnsrdos dnsscan httpscan httpsscan icmpscan icmpsdos memcachedrdos ntprdos ntpscan riprdos rstsdos sqlscan ssdprdos sshscan synsdos udpsdos"
LRSCAN="dns_lrscan http_lrscan icmp_lrscan netbios_lrscan rdp_lrscan smtp_lrscan snmp_lrscan ssh_lrscan telnet_lrscan vlc_lrscan"
MISC="sshpwdsm sshpwdmd sshpwdla telnetpwdsm telnetpwdmd telnetpwdla crossfiresm crossfiremd crossfirela lrtcpdos02 lrtcpdos05 lrtcpdos10 ackport ipidaddr ipidport"

for D in $DIMS; do
    echo ""
    echo "###################################################################"
    echo "#  KDE dim = $D"
    echo "###################################################################"
    echo ""

    # ── Step 1: Compute KDE features at this dimension ──────────────────
    KDE_DIR="$BASEDIR/kde_features_${D}"
    if [ -d "$KDE_DIR" ] && [ "$(ls -1 "$KDE_DIR"/*.csv 2>/dev/null | wc -l)" -ge 43 ]; then
        echo "[dim=$D] KDE features already exist in $KDE_DIR — skipping computation"
    else
        echo "[dim=$D] Computing KDE features (dim=$D) → $KDE_DIR"
        mkdir -p "$KDE_DIR"
        python "$BASEDIR/script/compute_kde_features_gpu.py" \
            --all-attacks \
            --data-dir "$BASEDIR/data" \
            --output-dir "$KDE_DIR" \
            --kde-dim "$D"
    fi

    # ── Step 2: Generate JSON configs ───────────────────────────────────
    echo "[dim=$D] Generating JSON configs"
    python "$BASEDIR/script/generate_kde_configs_dim.py" --kde-dim "$D"

    # ── Step 3: Run HyperVision for all 43 attacks ─────────────────────
    TEMP_DIR="$BASEDIR/temp_kde_${D}"
    CACHE_DIR="$BASEDIR/cache_kde_${D}"
    mkdir -p "$TEMP_DIR" "$CACHE_DIR"

    echo "[dim=$D] Running 43 attacks → $TEMP_DIR"

    for item in $BRUTE; do
        echo "  [dim=$D] brute/$item"
        ./HyperVision -config "$BASEDIR/configuration/bruteforce_kde_${D}/${item}.json" \
            > "$CACHE_DIR/${item}.log" 2>&1
    done

    for item in $LRSCAN; do
        echo "  [dim=$D] lrscan/$item"
        ./HyperVision -config "$BASEDIR/configuration/lrscan_kde_${D}/${item}.json" \
            > "$CACHE_DIR/${item}.log" 2>&1
    done

    for item in $MISC; do
        echo "  [dim=$D] misc/$item"
        ./HyperVision -config "$BASEDIR/configuration/misc_kde_${D}/${item}.json" \
            > "$CACHE_DIR/${item}.log" 2>&1
    done

    echo "[dim=$D] All 43 attacks complete"

    # ── Step 4: Analyze results ─────────────────────────────────────────
    echo "[dim=$D] Running analysis"
    cd "$ANALYZE_DIR"

    LOG_KDE_DIR="$BASEDIR/result_analyze/log_kde_${D}"
    mkdir -p "$LOG_KDE_DIR"

    # Backup real temp
    if [ -d "$BASEDIR/temp" ] && [ ! -L "$BASEDIR/temp" ]; then
        mv "$BASEDIR/temp" "$BASEDIR/temp_baseline_backup"
    fi

    # Symlink temp -> temp_kde_D
    ln -sfn "$TEMP_DIR" "$BASEDIR/temp"

    for g in brute lrscan misc; do
        mkdir -p "log/$g"
        rm -f "log/$g"/*.log
        mkdir -p "$LOG_KDE_DIR/$g"
        python batch_analyzer.py -g "$g" 2>&1 || true
        if [ -d log/$g ]; then
            cp -f log/$g/*.log "$LOG_KDE_DIR/$g/" 2>/dev/null || true
        fi
    done

    # Restore real temp
    rm -f "$BASEDIR/temp"
    if [ -d "$BASEDIR/temp_baseline_backup" ]; then
        mv "$BASEDIR/temp_baseline_backup" "$BASEDIR/temp"
    fi

    # Restore mutable log/ back to clean baseline snapshot for next iteration
    rm -rf "$ANALYZE_DIR/log"
    cp -a "$BASELINE_LOG_DIR" "$ANALYZE_DIR/log"

    # ── Step 5: Generate comparison CSV ─────────────────────────────────
    echo "[dim=$D] Generating comparison CSV"
    python "$BASEDIR/result_analyze/generate_comparison.py" \
        --baseline-dir "$BASELINE_LOG_DIR" \
        --kde-dir "$LOG_KDE_DIR" \
        --output "$BASEDIR/result_analyze/results_comparison_dim${D}.csv" \
        2>&1 || echo "[dim=$D] WARNING: generate_comparison.py failed (may need --kde-dir/--output flags)"

    cd "$BUILDDIR"
    echo "[dim=$D] ✓ Done"
done

echo ""
echo "###################################################################"
echo "#  Grid search complete for dims: $DIMS"
echo "#  Results: result_analyze/results_comparison_dim{D}.csv"
echo "#  Logs:    result_analyze/log_kde_{D}/"
echo "###################################################################"
