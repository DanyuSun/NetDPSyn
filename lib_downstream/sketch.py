import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import numpy as np
import hashlib
from collections import Counter
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config_dpsyn
from parameter_parser import parameter_parser
import logging

class CountSketch:
    def __init__(self, depth, width):
        self.depth = depth
        self.width = width
        self.sketch = np.zeros((depth, width))
        self.hash_functions = [self._hash_fn(i) for i in range(depth)]

    def _hash_fn(self, seed):
        def hash_fn(x):
            return int(hashlib.sha256((str(x) + str(seed)).encode()).hexdigest(), 16) % self.width
        return hash_fn

    def update(self, item, value):
        for i in range(self.depth):
            index = self.hash_functions[i](item)
            self.sketch[i][index] += value

    def query(self, item):
        estimates = []
        for i in range(self.depth):
            index = self.hash_functions[i](item)
            estimates.append(self.sketch[i][index])
        return np.median(estimates)

class CountMinSketch:
    def __init__(self, depth, width):
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width))
        self.hash_functions = [self._hash_fn(i) for i in range(depth)]

    def _hash_fn(self, seed):
        def hash_fn(x):
            return int(hashlib.sha256((str(x) + str(seed)).encode()).hexdigest(), 16) % self.width
        return hash_fn

    def update(self, item, value):
        for i in range(self.depth):
            index = self.hash_functions[i](item)
            self.table[i][index] += value

    def query(self, item):
        estimates = []
        for i in range(self.depth):
            index = self.hash_functions[i](item)
            estimates.append(self.table[i][index])
        return min(estimates)

class UnivMon:
    def __init__(self, depth, width, log_n):
        self.depth = depth
        self.width = width
        self.logn = log_n
        self.sketch = [CountSketch(depth, width // (2 ** i)) for i in range(log_n)]

    def _hash(self, flowkey, layer):
        return int(hashlib.sha256(flowkey.encode()).hexdigest(), 16) & 1

    def update(self, flowkey, val):
        for i in range(self.logn):
            if self._hash(flowkey, i):
                self.sketch[i].update(flowkey, val)
            else:
                break

    def query(self, flowkey):
        level = 0
        for level in range(self.logn):
            if not self._hash(flowkey, level):
                break
        level -= 1
        ret = self.sketch[level].query(flowkey)
        for i in range(level - 1, -1, -1):
            ret = 2 * ret - self.sketch[i].query(flowkey)
        return ret

class NitroSketch:
    def __init__(self, depth, width, threshold):
        self.depth = depth
        self.width = width
        self.tables = [Counter() for _ in range(depth)]
        self.threshold = threshold   

    def _hash(self, item, depth_level):
        return hash((item, depth_level)) % self.width

    def update(self, dstip):
        for i in range(self.depth):
            index = self._hash(dstip, i)
            self.tables[i][index] += 1

    def query(self, dstip):
        estimates = []
        for i in range(self.depth):
            index = self._hash(dstip, i)
            estimates.append(self.tables[i][index])
        return min(estimates)

def extract_dstip(data):
    return data['dstip'].values

def calculate_heavy_hitters_NitroSketch(sketch, data, total_packets):
    heavy_hitters = 0
    for dstip in set(data):
        if sketch.query(dstip) >= total_packets * sketch.threshold:
            heavy_hitters += 1
    return heavy_hitters

def calculate_heavy_hitters_UnivMon(univmon, data, threshold):
    counter = Counter()
    for item in data:
        counter[item] += 1
    total_count = sum(counter.values())
    heavy_hitters = sum(1 for item in counter if univmon.query(str(item)) >= threshold * total_count)
    return heavy_hitters

def cal_NitroSketch(real_path, synthetic_path):
    caida_real = pd.read_csv(real_path)
    caida_synthetic = pd.read_csv(synthetic_path)
 
    depth = 5  
    width = 100   
    threshold = 0.001   
    sketch_real = NitroSketch(depth, width, threshold)
    sketch_synthetic = NitroSketch(depth, width, threshold)

    for dstip in extract_dstip(caida_real):
        sketch_real.update(dstip)

    for dstip in extract_dstip(caida_synthetic):
        sketch_synthetic.update(dstip)
 
    total_packets_real = len(caida_real)
    total_packets_synthetic = len(caida_synthetic)
    heavy_hitters_real = calculate_heavy_hitters_NitroSketch(sketch_real, extract_dstip(caida_real), total_packets_real)
    heavy_hitters_synthetic = calculate_heavy_hitters_NitroSketch(sketch_synthetic, extract_dstip(caida_synthetic), total_packets_synthetic)
   
    error_real = heavy_hitters_real / total_packets_real
    error_syn = heavy_hitters_synthetic / total_packets_synthetic
    relative_error = abs(error_syn - error_real) / error_real if error_real != 0 else float('inf')

    # print('NitroSketch relative_error = ', relative_error)
    return relative_error

def cal_UnivMon(real_path, synthetic_path):
    caida_real = pd.read_csv(real_path)
    caida_synthetic = pd.read_csv(synthetic_path)

    dst_ips_real = caida_real['dstip']
    dst_ips_synthetic = caida_synthetic['dstip']
 
    univmon_real = UnivMon(depth=5, width=1000, log_n=10)
    univmon_synthetic = UnivMon(depth=5, width=1000, log_n=10)

    for dstip in dst_ips_real:
        univmon_real.update(str(dstip), 1)

    for dstip in dst_ips_synthetic:
        univmon_synthetic.update(str(dstip), 1)
 
    threshold = 0.001  
    heavy_hitters_real = calculate_heavy_hitters_UnivMon(univmon_real, dst_ips_real, threshold)
    heavy_hitters_synthetic = calculate_heavy_hitters_UnivMon(univmon_synthetic, dst_ips_synthetic, threshold)
 
    error_real = heavy_hitters_real / len(dst_ips_real)
    error_syn = heavy_hitters_synthetic / len(dst_ips_synthetic)
    relative_error = abs(error_syn - error_real) / error_real if error_real != 0 else float('inf')

    # print('UnivMon relative error = ', relative_error)
    return relative_error

def cms_cs(real_path, synthetic_path):
    cms_real = CountMinSketch(depth=5, width=1000)
    cms_synthetic = CountMinSketch(depth=5, width=1000)

    cs_real = CountSketch(depth=5, width=1000)
    cs_synthetic = CountSketch(depth=5, width=1000)

    caida_real = pd.read_csv(real_path)
    caida_synthetic = pd.read_csv(synthetic_path)

    dst_ips_real = caida_real['dstip']
    dst_ips_synthetic = caida_synthetic['dstip']
 
    for dstip in dst_ips_real:
        cms_real.update(str(dstip), 1)
        cs_real.update(str(dstip), 1)

    for dstip in dst_ips_synthetic:
        cms_synthetic.update(str(dstip), 1)
        cs_synthetic.update(str(dstip), 1)
 
    threshold = 0.001   
    hh_cms_real = calculate_heavy_hitters_UnivMon(cms_real, dst_ips_real, threshold)
    hh_cms_synthetic = calculate_heavy_hitters_UnivMon(cms_synthetic, dst_ips_synthetic, threshold)
    hh_cs_real = calculate_heavy_hitters_UnivMon(cs_real, dst_ips_real, threshold)
    hh_cs_synthetic = calculate_heavy_hitters_UnivMon(cs_synthetic, dst_ips_synthetic, threshold)
 
    error_cms_real = hh_cms_real / len(dst_ips_real)
    error_cms_syn = hh_cms_synthetic / len(dst_ips_synthetic)
    error_cs_real = hh_cs_real / len(dst_ips_real)
    error_cs_syn = hh_cs_synthetic / len(dst_ips_synthetic)
 
    relative_error_cms = abs(error_cms_syn - error_cms_real) / error_cms_real if error_cms_real != 0 else float('inf')
    relative_error_cs = abs(error_cs_syn - error_cs_real) / error_cs_real if error_cs_real != 0 else float('inf')
    
    # print('relative_error_cms = ', relative_error_cms)
    # print('relative_error_cs = ', relative_error_cs)
    
    return relative_error_cms, relative_error_cs

def show_results(raw, syn):
    cms, cs = cms_cs(raw, syn)
    univmon = cal_UnivMon(raw, syn)
    nitrosketch = cal_NitroSketch(raw, syn)
    result = {
            'CSM': [cms],
            'CS': [cs],
            'UnivMon':[univmon],
            'NitroSketch': [nitrosketch]
        }
    return result

def main(args):
    os.chdir("../../")
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
    file_prefix = args['dataset_name']
    print('Data:', file_prefix)
    raw_df = config_dpsyn.RAW_DATA_PATH + file_prefix + '.csv'
    syn_df = config_dpsyn.SYNTHESIZED_RECORDS_PATH + ('_'.join((args['dataset_name'], str(args['epsilon']))) + '.csv')
    result = show_results(raw_df, syn_df)
    print(result)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
