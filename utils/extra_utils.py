import random
import numpy as np
import torch


def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def get_best_model_name(modelname, args):
    # load the best model for the conditions
    pref = 'save/seed_%s/' % args['seed']
    best_model_filename = pref+'{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/best_%s_model.txt' % modelname
    with open(best_model_filename,'r') as f:
        best_model_name = f.readline()
    
    return pref+'{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/' + best_model_name
    