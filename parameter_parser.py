import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()
    
    ####################################### general parameters ###################################
    # parameters for single run
    parser.add_argument('--dataset_name', type=str, default="ton",
                        help='options: colorado')
    
    # parameters for workflow control
    parser.add_argument('--is_cal_marginals', type=str2bool, default=True)
    parser.add_argument('--is_cal_depend', type=str2bool, default=True)
    
    # parameters for privacy
    parser.add_argument('-e', '--epsilon', type=float, default=2.0,
                        help="when run main(), specify epsilon here")
    parser.add_argument('--depend_epsilon_ratio', type=float, default=0.1)
    parser.add_argument('--marg_add_sensitivity', type=float, default=1.0)
    parser.add_argument('--marg_select_sensitivity', type=float, default=4.0)
    parser.add_argument('--noise_add_method', type=str, default="A3",
                        help='A1 -> Equal Laplace; A2 -> Equal Gaussian; A3 -> Weighted Gaussian')
    
    # parameters for marginal selection
    parser.add_argument('--is_combine', type=str2bool, default=True)
    
    ############################################# specific parameters ############################################
    # parameters for view consist and non-negativity
    parser.add_argument('--non_negativity', type=str, default="N3",
                        help='N1 -> norm_cut; N2 -> norm_sub; N3 -> norm_sub + norm_cut')
    parser.add_argument('--consist_iterations', type=int, default=501, help='default value is 501')
    
    # parameters for synthesizing
    parser.add_argument('--initialize_method', type=str, default="singleton",
                        help='random -> randomized initial dataframe; singleton -> the distribution of each dataframe' 
                             'attribute matches noisy 1-way marginals; marginal_manual -> initialize the dataframe with mamually picked marginals;'
                             'marginal_auto -> initialize the dataframe with automatically picked marginals')
    parser.add_argument('--update_method', type=str, default="S5",
                        help='S1 -> all replace; S2 -> all duplicate; S3 -> all half-half;'
                             'S4 -> replace+duplicate; S5 -> half-half+duplicate; S6 -> half-half+replace.'
                             'The optimal one is S5')
    parser.add_argument('--append', type=str2bool, default=True)
    parser.add_argument('--sep_syn', type=str2bool, default=False)
    
    parser.add_argument('--update_rate_method', type=str, default="U4",
                        help='U4 -> step decay; U5 -> exponential decay; U6 -> linear decay; U7 -> square root decay.'
                             'The optimal one is U4')
    parser.add_argument('--update_rate_initial', type=float, default=1.0)
    parser.add_argument('--num_synthesize_records', type=int, default=int(6e5))
    parser.add_argument('--update_iterations', type=int, default=200)

    ############################################# parameters added by zl ############################################
    parser.add_argument('--dump_marginal', type=str, default='None',
                        help='dump one marginal by tuple of attr names to MARGINAL FOLDER')
    parser.add_argument('--is_syn_filtered', type=str2bool, default=False,
                        help='GUM on selected (not all) marginals, obsolete now')
    parser.add_argument('--pred_attr', type=str, default="type",
                        help='the attribute (typically label) that is predicted by downstream apps, only one is supported now,'
                             'type-> flow data'
                             'flag-> pcap data')

    ############################################# parameters added by ds ############################################
    parser.add_argument('--binning_method', type=str, default='manual',
                        help='manual -> manually add binning size;'
                             'optimized -> optimize binning method')

    parser.add_argument('--threshold', type=int, default=20000,
                        help='colorado -> 20000;'
                             'loan -> -20000;'
                             'accident -> -10000.0;'
                             'colorado-reduce -> 5000.0;'
                             'ton -> 20000;'
                             'ugr16 -> 20000;'
                             'cidds -> 200000;'
                             'caida->20000;')

    return vars(parser.parse_args())
