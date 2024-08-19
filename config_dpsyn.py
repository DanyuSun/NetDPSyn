# path related constant
RAW_DATA_PATH = "/Users/sdy/Desktop/dpsyn_clean_newest/temp_data/raw_data/"
PROCESSED_DATA_PATH = "/Users/sdy/Desktop/dpsyn_clean_newest/temp_data/processed_data/"
SYNTHESIZED_RECORDS_PATH = "/Users/sdy/Desktop/dpsyn_clean_newest/temp_data/synthesized_records/"
MARGINAL_PATH = "/Users/sdy/Desktop/dpsyn_clean_newest/temp_data/marginal/"
DEPENDENCY_PATH = '/Users/sdy/Desktop/dpsyn_clean_newest/temp_data/dependency/'

# RAW_DATA_PATH = "/home/dsun/dpsyn_clean_newest/temp_data/raw_data/"
# PROCESSED_DATA_PATH = "/home/dsun/dpsyn_clean_newest/temp_data/processed_data/"
# SYNTHESIZED_RECORDS_PATH = "/home/dsun/dpsyn_clean_newest/temp_data/synthesized_records/"
# MARGINAL_PATH = "/home/dsun/dpsyn_clean_newest/temp_data/marginal/"
# DEPENDENCY_PATH = '/home/dsun/dpsyn_clean_newest/temp_data/dependency/'

ALL_PATH = [RAW_DATA_PATH, PROCESSED_DATA_PATH, SYNTHESIZED_RECORDS_PATH, MARGINAL_PATH, DEPENDENCY_PATH]


# config file path
TYPE_CONIFG_PATH = "/Users/sdy/Desktop/dpsyn_clean_newest/fields.json"
#TYPE_CONIFG_PATH = "/home/dsun/dpsyn_clean_newest/fields.json"
# ZL: added srcip and dstip but not really working
MARGINAL_INIT = [('srcport', 'proto', 'flag'), ('dstport', 'proto', 'flag')]

#"[('dstport','type'), ('proto', 'byt', 'label', 'type'), ('ts', 'type'), ('td', 'type'), ('pkt', 'type'), ('srcport', 'type')]"