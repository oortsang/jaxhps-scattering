#!/bin/bash


python adaptive_meshing_accuracy_3D.py --hp_convergence --max_l 3 -p 8
python adaptive_meshing_accuracy_3D.py --hp_convergence --max_l 3 -p 12
python adaptive_meshing_accuracy_3D.py --hp_convergence --max_l 2 -p 16