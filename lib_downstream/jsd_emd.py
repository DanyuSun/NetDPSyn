import matplotlib.pyplot as plt
# Checking the columns available in the datasets
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_dpsyn
from parameter_parser import parameter_parser
# sys.path.append(config_dpsyn.PROJECT_PATH)

def calculate_jsd(df1, df2, column):
    count1 = df1[column].value_counts(normalize=True)
    count2 = df2[column].value_counts(normalize=True)
    merged_counts = pd.concat([count1, count2], axis=1, sort=False).fillna(0)
    jsd = jensenshannon(merged_counts.iloc[:, 0], merged_counts.iloc[:, 1])
    return jsd

def calculate_and_normalize_emd(df1, df2, continuous_fields):
    # Calculate EMD for continuous fields
    emd_values = {field: wasserstein_distance(df1[field], df2[field]) for field in continuous_fields}

    # Min and Max EMD values for normalization
    min_emd = min(emd_values.values())
    max_emd = max(emd_values.values())

    # Normalizing EMD values to [0.1, 0.9]
    normalized_emd = [0.1 + 0.8 * ((value - min_emd) / (max_emd - min_emd))
                      for field, value in emd_values.items()]

    #     normalized_emd = {f'EMD_{field}': 0.1 + 0.8 * ((value - min_emd) / (max_emd - min_emd))
    #                       for field, value in emd_values.items()}

    return normalized_emd


def calculate_jsd_and_normalized_emd(df1, df2, categorical_fields, continuous_fields):
    """
    Calculate the Jensen-Shannon Divergence (JSD) for categorical fields and
    the normalized Earth Mover's Distance (EMD) for continuous fields between two datasets.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param categorical_fields: List of categorical fields for JSD calculation
    :param continuous_fields: List of continuous fields for EMD calculation
    :return: Dictionary with JSD and normalized EMD values for each field
    """
    jsd_result = []
    results = {}

    # Calculate JSD for categorical fields
    for field in categorical_fields:
        jsd = calculate_jsd(df1, df2, field)
        jsd_result.append(jsd)
        # results[f'JSD_{field}'] = jsd

    # Calculate and normalize EMD for continuous fields
    normalized_emd_results = calculate_and_normalize_emd(df1, df2, continuous_fields)
    print(jsd_result, normalized_emd_results)
    return jsd_result, normalized_emd_results


def jsd_emd_flow(raw_data, synthetic_data):
    for field in ['srcip', 'dstip', 'srcport', 'dstport']:
        raw_data[field] = raw_data[field].astype('category')
        synthetic_data[field] = synthetic_data[field].astype('category')

    # Fields for JSD and EMD calculation
    jsd_fields = ['srcip', 'dstip', 'srcport', 'dstport', 'proto']
    emd_fields = ['ts', 'td', 'pkt', 'byt']

    categorical_fields = ['proto', 'type']
    continuous_fields = ['td', 'pkt', 'byt']

    # Calculating JSD and normalized EMD
    jsd_result, emd_result = calculate_jsd_and_normalized_emd(raw_data, synthetic_data, jsd_fields, emd_fields)
    return jsd_result, emd_result


def jsd_emd_pcap(raw_data, synthetic_data):
    raw_data['time_ms'] = raw_data['time'] / 1e6
    flow_size = raw_data.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto']).size().reset_index(name='flow_size')
    raw_data = raw_data.merge(flow_size, on=['srcip', 'dstip', 'srcport', 'dstport', 'proto'])

    synthetic_data['time_ms'] = synthetic_data['time'] / 1e6
    flow_size_2 = synthetic_data.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto']).size().reset_index(
        name='flow_size')
    synthetic_data = synthetic_data.merge(flow_size_2, on=['srcip', 'dstip', 'srcport', 'dstport', 'proto'])

    for field in ['srcip', 'dstip', 'srcport', 'dstport']:
        raw_data[field] = raw_data[field].astype('category')
        synthetic_data[field] = synthetic_data[field].astype('category')

    # Fields for JSD and EMD calculation
    jsd_fields = ['srcip', 'dstip', 'srcport', 'dstport', 'proto']
    emd_fields = ['ts', 'td', 'pkt', 'byt']

    categorical_fields = ['proto', 'type']
    continuous_fields = ['pkt_len', 'time_ms', 'flow_size']

    jsd_result = []
    emd_result = []

    # Calculate JSD for categorical fields
    for field in jsd_fields:
        jsd = calculate_jsd(raw_data, synthetic_data, field)
        jsd_result.append(jsd)

    emd_result =  calculate_and_normalize_emd(raw_data, synthetic_data, continuous_fields)
    return jsd_result, emd_result


if __name__ == "__main__":
    args = parameter_parser()
    file_prefix = args['dataset_name']

    raw_path = config_dpsyn.RAW_DATA_PATH + file_prefix + '.csv'
    netdpsyn_path = config_dpsyn.SYNTHESIZED_RECORDS_PATH + ('_'.join((args['dataset_name'], str(args['epsilon']))) + '.csv')
    print('read raw file:', raw_path)
    print('read synthesized file:', netdpsyn_path)

    if file_prefix in ['ton', 'ugr16','cidds']:
        jsd_netdpsyn, emd_netdpsyn = jsd_emd_flow(pd.read_csv(raw_path), pd.read_csv(netdpsyn_path))
        jsd_labels = ['SA', 'DA', 'SP', 'DP', 'PR']
        emd_labels = ['TS', 'TD', 'PKT', 'BYT']

        print("JSD Values:")
        for label, value in zip(jsd_labels, jsd_netdpsyn):
            print(f"{label}: {value}")
 
        print("\nEMD Values:")
        for label, value in zip(emd_labels, emd_netdpsyn):
            print(f"{label}: {value}")  

    
    elif file_prefix in ['caida', 'dc']:
        jsd_netdpsyn, emd_netdpsyn = jsd_emd_pcap(pd.read_csv(raw_path), pd.read_csv(netdpsyn_path))
        jsd_labels = ['SA', 'DA', 'SP', 'DP', 'PR']
        emd_labels = ['PS', 'PAT', 'FS']

        
        print("JSD Values:")
        for label, value in zip(jsd_labels, jsd_netdpsyn):
            print(f"{label}: {value}")
 
        print("\nEMD Values:")
        for label, value in zip(emd_labels, emd_netdpsyn):
            print(f"{label}: {value}")  
 