import argparse
import time
import random
import pandas as pd

from common import utils
from config import Config
import priors
import causal
import os

def parse_args():
    p = argparse.ArgumentParser(description="Run the causal discovery pipeline.")
    p.add_argument("--leg", type=str, help=f"{",".join(Config.SUPPLY_CHAIN_LEGS)}")
    return p.parse_args()

def run(experiment):
    print("="*50)
    print(f"Running Experiment : {experiment}")

    exp_dir = f'{Config.EXPERIMENTS_DIR}/{experiment}'

    # Load Data Dictionary
    data_dict_filepath = f'{exp_dir}/metadata/data_dictionary.json'
    data_dict = utils.read_json(data_dict_filepath)

    # Load Ref DAG pos
    pos_filepath =  f'{exp_dir}/metadata/ref_dag_pos.pkl'
    pos = utils.read_pickle(pos_filepath)

    # Load Ref DAG
    pos_filepath =  f'{exp_dir}/metadata/ref_dag.pkl'
    ref_dag = utils.read_pickle(pos_filepath)

    # Generate Priors
    prior_output_dir = f'{exp_dir}/priors'

    if not os.path.exists(prior_output_dir):
        utils.create_dir_if_not_exists(prior_output_dir)
        print("Generating priors .. ")
        priors.generate_priors(data_dict, prior_output_dir, ref_G=ref_dag) 

    # Load perfect_prior, random_priors, llm_priors, llm_consensus_prior
    all_priors = {}
    all_priors['perfect'] = utils.read_json(prior_output_dir + '/perfect_prior.json')
    all_priors['random'] = utils.read_json(prior_output_dir + '/random_priors_list.json')
    for llm, _ in Config.LLM.items():
        all_priors[llm] = utils.read_json(prior_output_dir + f'/{llm}_priors_list.json')
        all_priors[f'{llm}_consensus'] = utils.read_json(prior_output_dir + f'/{llm}_consensus_prior.json')


    # Evaluate Priors
    print("Evaluating priors")
    priors_summary, priors_all_eval = priors.generate_priors_report(data_dict, all_priors, ref_dag)
    utils.write_json(priors_summary, exp_dir + '/priors_summary.json')
    utils.write_json(priors_all_eval, exp_dir + '/priors_all.json')

    # Data Dir
    data_dir =  f'{exp_dir}/data'

    # Load Data Files List
    data_files_list_filepath = f'{exp_dir}/metadata/data_files_list.json'
    data_files_list = utils.read_json(data_files_list_filepath)

    # Create results dir
    results_output_dir =  f'{exp_dir}/results'
    utils.create_dir_if_not_exists(results_output_dir)

    all_discovery_priors = {}
    for key, value in all_priors.items():
        all_discovery_priors[key] = value if isinstance(value, dict) else value[random.randrange(len(value))]

    for data_file in data_files_list:
        data_filename = data_file['name']
        print(f"Processing {data_filename} .. ")
        df = pd.read_csv(data_dir + '/' + data_filename)

        # Changing random prior for each dataset
        all_discovery_priors['random'] = all_priors['random'][random.randrange(len(all_priors['random']))]
        discovery_results = causal.run_causal_discovery_pipeline(
                                                        df, 
                                                        data_dict, 
                                                        all_discovery_priors)

        # Evaluate individual runs
        for i in range(len(discovery_results)):
            discovery_results[i] = discovery_results[i] | causal.evaluate_graph(ref_dag, discovery_results[i]['dag'])
            discovery_results[i]['dag_json'] = utils.convert_graph_to_json(discovery_results[i]['dag'])
            del discovery_results[i]['dag']

        data_file['runs'] = discovery_results

        base, ext = os.path.splitext(data_filename)
        result_path = base + '_result.json'
        utils.write_json(data_file, results_output_dir + '/' + result_path)

    # Evaluate Causal Discovery
    result_files_list = os.listdir(results_output_dir)
    results = []
    for file in result_files_list:
        if "_result.json" in file:
            results.append(utils.read_json(results_output_dir + '/' + file))
    
    print("Writing report...")
    all_runs, summary = causal.generate_discovery_report(results)
    utils.write_json(all_runs, exp_dir + '/discovery_all_runs.json')
    utils.write_json(summary, exp_dir + '/discovery_summary.json')

    


if __name__ == "__main__":
    args = parse_args()

    print("Starting Causal Discovery pipeline...")
    if args.leg is not None:
        run(args.leg)
    else:
        for leg in Config.SUPPLY_CHAIN_LEGS:
            run(leg)

    print("Done.")
    

    


