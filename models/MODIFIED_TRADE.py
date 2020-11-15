import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
import json
# import pandas as pd
import copy

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
import pprint

from models.TRADE import *



class MODIFIED_TRADE(TRADE):
    
    def train_meta_batch(self, data_new, data_old, clip, slot_temp, slot_old, reset=0):
        
        # Make a deep copy of model parameters
        init_model_params = copy.deepcopy(self.state_dict())
        
        # Compute loss on the batch from the unseen domain
        self.train_batch(data_new, clip, slot_temp, reset=reset) # computes loss
        self.optimize(clip) # Calculates gradients and steps
        
        self.train_batch(data_old, clip, slot_old, reset=reset)
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        
        self.load_state_dict(init_model_params)
        self.optimizer.step()
        
    
#     def optimize(self, clip):
#         self.loss_grad.backward()
#         clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
#         self.optimizer.step()
#         
#         
#         
#         init_model_params = copy.deepcopy(student.model.state_dict())
#         
#         loss = student.compute_loss(batch)
#         student.backward(loss)
# 
#         optimizer.step()
#         
#         loss = student.compute_loss(batch_val)
#         student.backward(loss)
#         
#         student.model.load_state_dict(init_model_params)
#         optimizer.step()
        
        
        
        