import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
import sys
sys.path.append('/home/dsun/NetDPSyn/')
import config_dpsyn
import os
import logging
from parameter_parser import parameter_parser


def main(args):
    os.chdir("../../")
    
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    file_prefix = args['dataset_name']

    # You can add dataset here to make Comparison: 
    datasets = {
        file_prefix + 'Raw': config_dpsyn.RAW_DATA_PATH + file_prefix + '.csv',
        file_prefix + 'Syn': config_dpsyn.SYNTHESIZED_RECORDS_PATH + ('_'.join((args['dataset_name'], str(args['epsilon']))) + '.csv'),

    }

    plt.set_loglevel(level = 'warning')

    # Prepare figure
    plt.figure(figsize=(10, 6))

    # Labels and titles
    plt.xlabel('Time lag (seconds)')
    plt.ylabel('Autocorrelation')
    plt.title('Comparison of Autocorrelation for Different Datasets')
    plt.ylim(-0.1, 0.5)

    # Define different markers for each dataset for the legend
    markers = ['o', 's', '^']  # circle, square, triangle_up
    colors = ['b', 'g', 'r']  # blue, green, red
    i = 0
    # Compute and plot ACF for each dataset
    for dataset_name, data_path in datasets.items():
        # Convert 'ts' to datetime in seconds
        print("compute acf on " + dataset_name)
        dataset = pd.read_csv(data_path)
        dataset['ts'] = pd.to_datetime(dataset['ts'], unit='us')
        
        # Set the timestamp as the index of the dataframe
        dataset.set_index('ts', inplace=True)
        
        # Resample the data to a one-second interval and sum the bytes
        secondly_data = dataset['byt'].resample('S').sum().dropna()
        
        # Compute autocorrelation for the one-second interval resampled data
        autocorrelation_secondly = acf(secondly_data, fft=True, nlags=100)
        print(autocorrelation_secondly)
        # Generate the lag values for manual plotting
        lags = np.arange(0, len(autocorrelation_secondly))
        
        # Plot the ACF for this dataset
        #plt.stem(lags, autocorrelation_secondly, markerfmt='o', linefmt=f'C{i}-', basefmt=" ", label='dataset_name')
        plt.stem(lags, autocorrelation_secondly, linefmt=f'{colors[i]}-', markerfmt=f'{colors[i]}{markers[i]}', basefmt=" ", label=dataset_name)
        i = i + 1
    # Add a legend to distinguish the datasets
    plt.legend()

    plt.show()

if __name__ == "__main__":
    args = parameter_parser()
    
    main(args)

