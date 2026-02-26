#!/bin/bash


tasks=(
    "CloseBlenderLid"
    "CloseToasterOvenDoor"
    "OpenCabinet"
    "OpenDrawer"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "PickPlaceSinkToCounter"
    "SlideDishwasherRack"
    "TurnOnElectricKettle"
    "TurnOnSinkFaucet"
)

# concat tasks with +
tasks_str=$(IFS=+; echo "${tasks[*]}")


python prepare_json_robocasa.py \
  --data_root data \
  --dataset_names "${tasks_str}" \
  --out_dir data/annotation_all \
  --sequence_length 8 \
  --sequence_interval 1 \
  --start_interval 1

