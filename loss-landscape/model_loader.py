import os
import dataset.model_loader as loader

def load(args, dataset, model_name, model_file, data_parallel=False):
    net = loader.load(args, model_name, model_file, data_parallel)
    return net
