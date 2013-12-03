
from msmbuilder import arglib
from msmbuilder import metrics
import numpy as np
import pickle

parser = arglib.ArgumentParser(get_basic_metric=True)
parser.add_argument('tau', default=1, type=int, help='tau to use when calculating the drift at each conformation')
parser.add_argument('output', default='drift_metric.pickl', help='output filename to pickle the metric to.')

args, metric = parser.parse_args()

drift_metric = metrics.DriftMetric(metric, args.tau)

with open(args.output, 'wb') as fh:
    pickle.dump(drift_metric, fh)

print "saved output to %s" % args.output
