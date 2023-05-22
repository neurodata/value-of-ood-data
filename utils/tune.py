import wandb
import numpy as np
from copy import deepcopy

from utils.run_net import train, evaluate

def search_alpha(cfg, prev_alpha, net, dataloaders):

    # adaptive search space
    if prev_alpha == 1:
        alpha_range = np.arange(0.5, 1+1e-5, 0.025)[:-1]
    else:
        alpha_range = np.linspace(prev_alpha, 1, 20)[:-1]

    scores = []
 
    for alpha in alpha_range:
        cfg.loss.alpha = np.float64(alpha).item()
        tune_net = deepcopy(net) # deepcopy the network architecture
        train(cfg, tune_net, dataloaders)
        err =  evaluate(cfg, tune_net, dataloaders[1], 0)
    
        info = {
            "alpha": round(alpha, 4),
            "error_at_alpha": round(err, 4)
        }
        print(info)

        if cfg.deploy:
            wandb.log(info)

        scores.append(err)

    return alpha_range[np.argmin(scores)], scores[np.argmin(scores)]


    



