# path related constant
RAW_DATA_PATH = "temp_data/raw_data/"
PROCESSED_DATA_PATH = "temp_data/processed_data/"
SYNTHESIZED_RECORDS_PATH = "temp_data/synthesized_records/"
MARGINAL_PATH = "temp_data/marginal/"
DEPENDENCY_PATH = 'temp_data/dependency/'

ALL_PATH = [RAW_DATA_PATH, PROCESSED_DATA_PATH, SYNTHESIZED_RECORDS_PATH, MARGINAL_PATH, DEPENDENCY_PATH]

# config file path
TYPE_CONIFG_PATH = "Net-PrivSyn/dpsyn_clean/fields.json"
MARGINAL_INIT = "[('dstport','type'), ('proto', 'byt', 'label', 'type'), ('ts', 'type'), ('td', 'type'), ('pkt', 'type'), ('srcport', 'type')]"
