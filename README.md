rembrandt (in honour of Rembrandt Harmenszoon van Rijn)
synthetic computer vision dataset creation made easy.

generate training data for object detection from single `.obj` file:
- randomized camera angles, lighting, backgrounds and 2D augmentations
- output directly in YOLO format with a ready-to-run training script

early stage development. not yet usable.

## camera pose preview

Install the preview extras and launch the interactive browser view:

```bash
pip install -e ".[preview]"
streamlit run scripts/camera_pose_preview.py
```

Use the sliders to tune `sample_camera_poses` values, then copy the generated
snippet into `src/rembrandt/main.py` or a future config file.

copyright @conchiglia 2026