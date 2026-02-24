#!/bin/bash


tasks=(
    "OpenCabinet" 
    "CloseBlenderLid"
    "CloseDrawer"
    "OpenDrawer"
    "PickPlaceCounterToCabinet"
    "PickPlaceStoveToCounter"
    "SlideDishwasherRack"
    "CloseCabinet"
    "PickPlaceCabinetToCounter"
    "PickPlaceCounterToSink"
    "PickPlaceToasterToCounter"
)

# concat tasks with +
tasks_str=$(IFS=+; echo "${tasks[*]}")


python prepare_json_robocasa.py \
  --data_root data \
  --dataset_names "${tasks_str}" \
  --out_dir data/annotation_all \
  --sequence_length 8 \
  --sequence_interval 2 \
  --start_interval 1

