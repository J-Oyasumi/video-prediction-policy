#!/bin/bash
#SBATCH -J vpp_prepare_latent
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=b200-mig45

ml load cuda/12.8

eval "$(conda shell.bash hook)"
conda activate vpp

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

# target tasks
target_tasks=(
    "OpenCabinet" 
    "CloseBlenderLid"
    "CloseDrawer" #
    "OpenDrawer"
    "PickPlaceCounterToCabinet"
    "PickPlaceStoveToCounter" #
    "SlideDishwasherRack"
    "CloseCabinet" #
    "PickPlaceCabinetToCounter" #
    "PickPlaceCounterToSink" #
    "PickPlaceToasterToCounter"

    "CloseFridge" #
    "CloseToasterOvenDoor"
    "CoffeeSetupMug"
    "OpenStandMixerHead" #
    "PickPlaceCounterToStove" # small std
    "PickPlaceDrawerToCounter" # 
    "PickPlaceSinkToCounter"
    "TurnOffStove"
    "TurnOnElectricKettle"
    "TurnOnMicrowave"
    "TurnOnSinkFaucet"
)


for task in ${tasks[@]}; do
    python prepare_latent_robocasa.py --task_name $task
done