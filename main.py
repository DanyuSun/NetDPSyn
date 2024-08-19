import logging
#import mkl
import os
import time
from exp.exp_dpsyn_gum import ExpDPSynGUM
from parameter_parser import parameter_parser
from lib_preprocess.preprocess_network import PreprocessNetwork


def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main(args):
    # config the logger
    config_logger()
    os.chdir("../../")
    ExpDPSynGUM(args)

    preprocess = PreprocessNetwork()

    synthesized_filename = '_'.join((args['dataset_name'], str(args['epsilon']), str(args['update_iterations']), args['initialize_method']))
    mapping_filename = args['dataset_name'] + '_mapping'
    #gaussian_filename = args['dataset_name'] + '_gaussian'

    #ZL TBD: release a noisy version of gaussian
    #preprocess.reverse_mapping_from_files(synthesized_filename, mapping_filename, gaussian_filename)
    preprocess.reverse_mapping_from_files(synthesized_filename, mapping_filename)
    preprocess.save_data_csv(synthesized_filename + '.csv')


if __name__ == "__main__":
    print('----start main -----')
    print(time.time())
    args = parameter_parser()
    
    main(args)

    print('----end main -----')
    print(time.time())
