#!/bin/bash
eval "$($(which conda) 'shell.bash' 'hook')"
echo "Activating conda environment"
# conda activate ml_env

echo "Converting annotations to masks"
python maskGen.py

conda deactivate 
