import torch

def federated_average(models):
    global_model = models[0]
    global_dict = global_model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = torch.stack([model.state_dict()[k].float() for model in models]).mean(0)
        
    global_model.load_state_dict(global_dict)
    return global_model
