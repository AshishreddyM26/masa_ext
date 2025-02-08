# masa_custom_model
Use MASA with custom trained detection/segmentation model.

## Installation
#### Please refer to [INSTALL.md](install.md)

## Demo

### 1. Import MASA Tracker
```sh
from main import TrackerMASA
masa_tracker = TrackerMASA()
```
### 2. Pass the Req
```sh
# replace placeholders with your own paths

video_root = "path/to/your/video.mp4"
video_sink = "path/to/out_video.mp4"

weights = "best.pt"

masa_config = "masa/configs/masa-one/masa_r50_plug_and_play.py"
masa_checkpoint = "masa/saved_models/masa_models/masa_r50.pth"
```

### 3. Run the MASA Tracker
```sh
masa_tracker.run_masa_tracking(video_root, video_sink, weights, masa_config, masa_checkpoint)

```


### Official Citation 

```bibtex
@article{masa,
  author    = {Li, Siyuan and Ke, Lei and Danelljan, Martin and Piccinelli, Luigi and Segu, Mattia and Van Gool, Luc and Yu, Fisher},
  title     = {Matching Anything By Segmenting Anything},
  journal   = {CVPR},
  year      = {2024},
}
