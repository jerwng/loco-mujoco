This directory contains legacy scenarios for lora finetune tuning.

It is deprecated due to several issues with the dataset, including:
- Naming scenarios with object name `_occluders`, which is not supported by the `annotate_scene_bboxes` script.
- Items are fully ocluded behind another object. This is not a supported scenario.