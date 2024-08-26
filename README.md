# [IMC'24] NetDPSyn: Synthesizing Network Traces under Differential Privacy

## Introduction

This repository contains code for the paper: NetDPSyn: Synthesizing Network Traces under Differential Privacy. NetDPSyn is the first system to synthesize high-fidelity network traces under privacy guarantees.

## Experimental setup
1. Clone this repo from the GitHub.

        git clone https://github.com/DanyuSun/NetDPSyn.git

1. Download the raw datasets from [here](https://drive.google.com/drive/folders/1MHRJxLhnJWZln8XBCon9UrN_EwVj14BE). And save them in `./temp_data/raw_data/` folder.

2. Your directory structure should look like this:

        NetDPSyn
           └── temp_data
                   └── raw_data
                        └── caida.csv
                        └── cidds.csv
                        └── dc.csv
                        └── ton.csv
                        └── ugr16.csv
           └── exp
           └── ...

4. **Note:** Please update all the paths in `config_dpsyn.py` to match your local directory structure.


## Usage

1. **Preprocess Data**. Run `lib_preprocess/preprocess_network.py`. This will generate a preprocessed pickle file in the `temp_data/processed_data` folder, along with a mapping for binning. Additionally, a trivially decoded CSV file (binning and unbinning) will be created in the `temp_data/synthesized_records` folder.

        python3 preprocess_network.py


2. **Synthesize Data**. Next, run `main.py` to generate the synthesized data. The synthesized data will be saved in the `temp_data/synthesized_records` floder.

        python3 main.py


3. **Downstream Tasks**. You can run code from `lib_downstream` (e.g., `lib_downstream/flow_eval.py`). This will print out the evaluation results for both the raw dataset and the synthesized dataset.


## Citation
If you find our work useful for your research, please consider citing the paper: Coming Soon.


