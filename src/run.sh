#!/bin/bash
# conda activate pytorch
config_directory="src/config_run"

# Iterate over each config file in the directory
for config_file in "$config_directory"/*.py; do
    if [ "$(basename "$config_file")" = "empty.py" ]; then
        continue
    fi
    echo "Running with config file: $config_file"
    python src/pl_trainevaltestsave.py $config_file
    echo "-------------------------------------"
done