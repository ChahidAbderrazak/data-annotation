#!/bin/bash
eval "$($(which conda) 'shell.bash' 'hook')"
clear
env_name=xct-cpu-env
########################   Ubuntu-CPU   ########################
echo "Activating conda environment"
conda activate $env_name
echo "Running python script on Ubuntu"

#|||||||||||||||||  Annotation functions    |||||||||||||||||
python lib/annotate_segmentation_with_text-napari.py


#|||||||||||||||||  Annotation GUI    |||||||||||||||||
# python main.py 

conda deactivate 
