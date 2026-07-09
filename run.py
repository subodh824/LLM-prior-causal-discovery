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
    print("Reading priors if exists..")
    prior_output_dir = f'{exp_dir}/priors'
    utils.create_dir_if_not_exists(prior_output_dir)

    # Load perfect_prior, random_priors, llm_priors, llm_consensus_prior
    perfect_prior = utils.read_json(prior_output_dir + '/perfect_prior.json')
    random_priors = utils.read_json(prior_output_dir + '/random_priors_list.json')
    llm_priors = utils.read_json(prior_output_dir + '/llm_priors_list.json')
    llm_consensus_prior = utils.read_json(prior_output_dir + '/llm_consensus_prior.json')

    if not perfect_prior or not random_priors or not llm_priors or not llm_consensus_prior:
        print("Generating Missing priors .. ")
        priors.generate_priors(data_dict, prior_output_dir, ref_G=ref_dag) 

    # Evaluate Priors
    print("Evaluating priors")
    priors_summary, priors_result  = priors.generate_priors_report(data_dict, perfect_prior, random_priors, llm_priors, ref_dag)
    utils.write_json(priors_summary, exp_dir + '/priors_summary.json')
    utils.write_json(priors_result, exp_dir + '/priors_report.json')

    # Data Dir
    data_dir =  f'{exp_dir}/data'

    # Load Data Files List
    data_files_list_filepath = f'{exp_dir}/metadata/data_files_list.json'
    data_files_list = utils.read_json(data_files_list_filepath)

    # Create results dir
    results_output_dir =  f'{exp_dir}/results'
    utils.create_dir_if_not_exists(results_output_dir)

    for data_file in data_files_list:
        data_filename = data_file['name']
        print(f"Processing {data_filename} .. ")
        df = pd.read_csv(data_dir + '/' + data_filename)

        discovery_results = causal.run_causal_discovery_pipeline(
                                                        df, 
                                                        data_dict, 
                                                        perfect_prior, 
                                                        random_priors[random.randrange(len(random_priors))], 
                                                        llm_consensus_prior)

        # Evaluate individual runs
        for i in range(len(discovery_results)):
            discovery_results[i] = discovery_results[i] | causal.evaluate_graph(ref_dag, discovery_results[i]['dag'])

        data_file['runs'] = discovery_results

        base, ext = os.path.splitext(data_filename)
        result_path = base + '_result.pkl'
        utils.write_pickle(data_file, results_output_dir + '/' + result_path)

    # Evaluate Causal Discovery
    result_files_list = os.listdir(results_output_dir)
    results = []
    for file in result_files_list:
        if "_result.pkl" in file:
            results.append(utils.read_pickle(results_output_dir + '/' + file))
    
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
    

    


