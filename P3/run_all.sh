#!/bin/bash
set -e

echo "========================================================="
echo "  Executing All Image Processing Tasks (Docker Mode)     "
echo "=========================================================\n"

echo ">>> RUNNING TASK 1: Chemical Noise Removal..."
python task1_chemical_noise.py
echo -e "\n---------------------------------------------------------\n"

echo ">>> RUNNING TASK 2: Speckle Noise Removal..."
python task2_speckle_removal.py
echo -e "\n---------------------------------------------------------\n"

echo ">>> RUNNING TASK 3: MRI DICOM Visualisation..."
python task3_mri_visualization.py
echo -e "\n---------------------------------------------------------\n"

echo "All tasks completed successfully! Check the mapped 'output/' directory."
