import os
import torch, torchvision
import dataset.models.vgg as vgg
import dataset.models.resnet as resnet
import dataset.models.densenet as densenet
import json
import dataset.models.fcnn as fcnn


# map between model name and function
models = {
    'vgg9'                  : vgg.VGG9,
    'densenet121'           : densenet.DenseNet121,
    'resnet18'              : resnet.ResNet18,
    'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34'              : resnet.ResNet34,
    'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50'              : resnet.ResNet50,
    'resnet50_noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    'resnet101_noshort'     : resnet.ResNet101_noshort,
    'resnet152'             : resnet.ResNet152,
    'resnet152_noshort'     : resnet.ResNet152_noshort,
    'resnet20'              : resnet.ResNet20,
    'resnet20_noshort'      : resnet.ResNet20_noshort,
    'resnet32_noshort'      : resnet.ResNet32_noshort,
    'resnet44_noshort'      : resnet.ResNet44_noshort,
    'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
    'resnet56'              : resnet.ResNet56,
    'resnet56_noshort'      : resnet.ResNet56_noshort,
    'resnet110'             : resnet.ResNet110,
    'resnet110_noshort'     : resnet.ResNet110_noshort,
    'wrn56_2'               : resnet.WRN56_2,
    'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    'wrn56_4'               : resnet.WRN56_4,
    'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    'wrn56_8'               : resnet.WRN56_8,
    'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
}

def get_total_params_function(depth, units):
    total_params = 0
    
    for layer in range(1, depth + 2):
        params = units[layer] * ( units[layer - 1] + 1 )
        total_params += params
    
    return total_params

def get_units_equal(input_size, output_size, depth, params_limit=15500):
    units = [input_size]
    possible_num_of_unit = 0
    test = 1
    
    while True:
        test_units = [input_size] + [test] * depth + [output_size]
        test_params = get_total_params_function(depth, test_units)
        if test_params > params_limit:
            break
        else:
            possible_num_of_unit = test
            test += 1
            
    units.extend([possible_num_of_unit] * depth)
    units.append(output_size)
    return units

def load(args, model_name, model_file=None, data_parallel=False):
    # 수정
    if model_name == "fcnn":
        if args.unit_equal:
            units = get_units_equal(args.input_size, args.output_size, args.depth, args.param_limit + 500)
        else:
            units = json.loads(args.units)
        net = fcnn.CustomFCNN(args.depth, units, args.input_size, args.output_size)
    else:
        net = models[model_name]()
        
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        model_pth = torch.load(model_file)
        if 'state_dict' in model_pth.keys():
            net.load_state_dict(model_pth['state_dict'])
        else:
            net.load_state_dict(model_pth)
        # net.state_dict()
        
    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
