from config import Config
import json
import os
import time

from discovery import pc
from discovery import hill_climb

from common import utils

def run_causal_discovery_pipeline(df, data_dict, perfect_prior, random_prior, llm_prior, output_dir):

    prev = time.time()

    results = []

    # ===== PC
    print("Running PC Baseline")
    results.append({ 
        'algo': 'pc_baseline',
        'dag': pc.baseline(df, 0.05)
    })

    print("Running PC with perfect priors")
    results.append({ 
        'algo': 'pc_perfect_prior',
        'dag': pc.with_priors(df, 0.05, perfect_prior)
    })

    print("Running PC with random priors")
    results.append({ 
        'algo': 'pc_random_prior',
        'dag': pc.with_priors(df, 0.05, random_prior)
    })

    print("Running PC with LLM priors")
    results.append({ 
        'algo': 'pc_llm_prior',
        'dag': pc.with_priors(df, 0.05, llm_prior)
    })

    # ======= Hill Climb 
    print("Running HillClimb Baseline")
    results.append({ 
        'algo': 'hc_baseline',
        'dag': hill_climb.baseline(df)
    })

    lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("Running HillClimb with perfect priors")
    for lamb in lambda_grid:
        results.append({ 
            'algo': 'hc_perfect_prior',
            'dag': hill_climb.with_priors(df, perfect_prior, lamb),
            'params': {
                'lambda': lamb
            }
        })

    print("Running HillClimb with random priors")
    for lamb in lambda_grid:
        results.append({ 
            'algo': 'hc_random_prior',
            'dag': hill_climb.with_priors(df, random_prior, lamb),
            'params': {
                'lambda': lamb
            }
        })
    
    print("Running HillClimb with LLM priors")
    for lamb in lambda_grid:
        results.append({ 
            'algo': 'hc_llm',
            'dag': hill_climb.with_priors(df, llm_prior, lamb),
            'params': {
                'lambda': lamb
            }
        })
    
    print(f"Took {time.time()-prev:.1f} secs\n")
    
    # utils.write_pickle(results, output_dir + 'results.pkl')
    return results


