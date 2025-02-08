import logging
import gc
# import resource
import argparse
import cv2
from tqdm import tqdm

import sys
import os
import time
import numpy as np

import supervision as sv
from ultralytics import YOLO
import torch
from torch.nn import functional as F
import torch
from torch.multiprocessing import Pool, set_start_method

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms
from mmdet.structures import DetDataSample

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry

import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import hf_hub_download
import joblib
import pickle


class TrackerMASA():

    def run_masa_tracking(self, video_root, video_sink, weights, masa_config, masa_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(weights).to(device)

        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator(thickness=1)
        track_annotator = sv.TraceAnnotator()
        label_annotator = sv.LabelAnnotator()

        video_info = sv.VideoInfo.from_video_path(video_path=video_root)
        frame_generator = sv.get_video_frames_generator(source_path=video_root)

        # Init the MASA tracker model
        masa_model = init_masa(masa_config, masa_checkpoint, device='cuda')
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)

        frame_idx = 1

        with sv.VideoSink(video_sink, video_info=video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                # Do the detections
                result = model(frame, imgsz=960, conf=.3, verbose=False)[0]

                # Reformat the ultralytics results into detections for MASA
                det_bboxes = result.boxes.cpu().xyxy
                det_scores = result.boxes.cpu().conf
                det_labels = result.boxes.cpu().cls

                det_bboxes = torch.cat([det_bboxes, det_scores.unsqueeze(1)], dim=1)

                # Do the tracking!
                track_result, fps = inference_masa(masa_model, frame, frame_id=frame_idx,
                                                video_len=video_info.total_frames,
                                                test_pipeline=masa_test_pipeline,
                                                det_bboxes=det_bboxes,
                                                det_labels=det_labels,
                                                fp16=True,
                                                show_fps=True)

                # Get the most recent frame of tracked objects
                track_result = track_result.video_data_samples[-1]

                # Create a new MMDet object and put the data from the TrackDataSample object into the new DetDataSample so Supervision can process it.
                tracks = DetDataSample()
                pred_instances = track_result.pred_track_instances
                tracks.pred_instances = pred_instances

                # Add the tracks to supervision
                detections = sv.Detections.from_mmdetection(tracks)
                # Add the tracker_ids separately (supervision doesn't support TrackDataSample objects yet)
                detections.tracker_id = tracks.pred_instances.instances_id.cpu().numpy()

                labels = [
                    f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, tracker_id, _ in detections
                ]

                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections)
                annotated_frame = track_annotator.annotate(
                    scene=annotated_frame.copy(),
                    detections=detections)
                
                try:
                    annotated_frame = mask_annotator.annotate(
                        scene=annotated_frame.copy(),
                        detections=detections
                    )
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame.copy(),
                        labels=labels,
                        detections=detections
                    )

                    sink.write_frame(frame=annotated_frame)

                    frame_idx += 1

                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                    raise
