#!/bin/bash

CONFIG_DIR="configs/task_arithmetic/gen"
PYTHON_SCRIPT="scripts/task_arithmetic/ta_generate.py"

for config_file in "$CONFIG_DIR"/*; do
    if [[ -f "$config_file" ]]; then
        echo "Running config: $config_file"
        python "$PYTHON_SCRIPT" --config "$config_file"
    fi
done
