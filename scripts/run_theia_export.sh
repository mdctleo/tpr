#!/bin/bash
# Wrapper script to run the THEIA dataset export with proper environment
# 
# Usage:
#   ./scripts/run_theia_export.sh
#
# This script:
# 1. Loads the mamba module
# 2. Activates the pids environment
# 3. Starts the database (if not already running)
# 4. Runs the export script

set -e

cd /scratch/asawan15/PIDSMaker

echo "========================================"
echo "THEIA Dataset Export Wrapper"
echo "========================================"

# Load environment
echo "Loading mamba module..."
module load mamba

echo "Activating pids environment..."
source activate pids

# Check if database is running
echo "Checking database status..."
if ! pg_isready -h localhost -p 5432 -U postgres > /dev/null 2>&1; then
    echo "Database not running, starting it..."
    cd /scratch/asawan15/PIDSMaker/scripts/apptainer
    make up
    cd /scratch/asawan15/PIDSMaker
    sleep 5
else
    echo "Database is already running."
fi

# Run the export script
echo "Starting export..."
python scripts/export_theia_datasets.py

echo "========================================"
echo "Export complete!"
echo "========================================"
