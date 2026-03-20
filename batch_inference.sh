#!/bin/bash
python inference.py --name "GeneratedMesh_dress" --orth_scale 1.35 --output_dir InferenceResults/
python inference.py --name "GeneratedMesh_shirt" --orth_scale 1.25 --output_dir InferenceResults/
python inference.py --name "GeneratedMesh_skirt" --orth_scale 1.5 --output_dir InferenceResults/
python inference.py --name "GeneratedMesh_skirt2" --orth_scale 1.5 --output_dir InferenceResults/
python inference.py --name "GeneratedMesh_sleep_dress" --orth_scale 1.5 --output_dir InferenceResults/