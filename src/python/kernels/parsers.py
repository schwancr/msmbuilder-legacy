
from msmbuilder.metrics.parsers import add_argument, add_basic_metric_parsers, construct_basic_metric
from msmbuilder.kernels import DotProduct

def add_kernel_parsers(parser):
    
    kernel_parser_list = []
    kernel_subparser = parser.add_subparsers(dest='kernel', description='Available kernels to use.')


    kernel_parser_list.extend( add_layer_kernel_parsers(kernel_subparser) )

def add_layer_kernel_parsers(kernel_subparser):

    kernel_parser_list = []

    dotproduct = kernel_subparser.add_parser('dotproduct', description='''
        Use the dot product between vectors as your kernel function. NOTE: By using this kernel you
        are not actually using the 'kernel trick' since we are explicitly calculating the feature space''')

    dotproduct_subparsers = dotproduct.add_subparsers(dest='sub_metric', description='''
        Need to make a vectorized version of the protein conformation in order to use a kernel function.''')

    dotproduct.metric_parser_list = add_basic_metric_parsers(dotproduct_subparsers)

    kernel_parser_list.extend(dotproduct.metric_parser_list)

    return kernel_parser_list

def construct_layer_kernel(kernel_name, args):
    
    if kernel_name == 'dotproduct':
        sub_metric = construct_basic_metric(args.sub_metric, args)
    
        return DotProduct(sub_metric)
