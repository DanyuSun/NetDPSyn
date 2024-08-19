# [IMC'24] NetDPSyn: Synthesizing Network Traces under Differential Privacy

## Introduction

This repository contains code for the paper: NetDPSyn: Synthesizing Network Traces under Differential Privacy. NetDPSyn is the first system to synthesize high-fidelity network traces under privacy guarantees.

## Experimental setup

1. Download the raw datasets from [here](https://drive.google.com/drive/folders/1MHRJxLhnJWZln8XBCon9UrN_EwVj14BE). There is also a ton.csv dataset in `temp_data/raw_data`. Move this folder to `../../` (one level above `Net-PrivSyn`, assuming you check out this parent repo).

2. Install the required packages if you haven't done so, including  `numpy, pandas, scipy, networkx, scikit-learn, matplotlib`.

3. If you're using VSCode, you can change `.vscode/launch.json` to select which code to run under `program`. It also has the `args` defined.

## Usage

1. You first need to preprocess data. Run `lib_preprocess/preprocess_network.py`, set `dataset_name` of `args` to the raw dataset file (without the `.csv` extension, like `ton`). The `num_synthesize_records` can be set to the similar number of the raw dataset. A preprocessed pickle file will be generated, under `temp_data/processed_data`, together with a mapping for binning. It will also generate a trivially decoded csv (binning and unbinning) in `temp_data/synthesized_records`.

2. Then, you need to run `main.py` to generate synthesized data with privsyn, which will be under `temp_data/synthesized_records`.

3. For downstream tasks, you can run code from `lib_downstream` (e.g., `lib_downstream/flow_eval.py`) and specify the same `dataset_name`. It prints out the evaluation results on raw dataset and synthesized dataset.


## Citation
If you find our work useful for your research, please consider citing the paper: Coming Soon.


