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
    p.add_argument("--data", type=str, help="Data dir ")
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
    print("Generating priors ..")
    prior_output_dir = f'{exp_dir}/priors'
    utils.create_dir_if_not_exists(prior_output_dir)

    priors.generate_priors(data_dict, prior_output_dir, ref_G=ref_dag) 

    # Load perfect_prior, random_priors, llm_priors, llm_consensus_prior
    perfect_prior = utils.read_json(prior_output_dir + '/perfect_prior.json')
    random_priors = utils.read_json(prior_output_dir + '/random_priors_list.json')
    llm_priors = utils.read_json(prior_output_dir + '/llm_priors_list.json')
    llm_consensus_prior = utils.read_json(prior_output_dir + '/llm_consensus_prior.json')

    # Evaluate Priors
    print("Evaluating priors")
    summary, all_priors_eval  = priors.generate_priors_report(data_dict, perfect_prior, random_priors, llm_priors, ref_dag)
    utils.write_json(summary, exp_dir + '/priors_summary.json')
    utils.write_json(all_priors_eval, exp_dir + '/priors_report.json')

    # Load data files list
    data_dir =  f'{exp_dir}/data'
    data_files_list = os.listdir(data_dir)

    # Create results dir
    results_output_dir =  f'{exp_dir}/results'
    utils.create_dir_if_not_exists(results_output_dir)

    for data_file in data_files_list:
        print(f"Processing {data_file} .. ")
        df = pd.read_csv(data_dir + '/' + data_file)

        result = causal.run_causal_discovery_pipeline(df, data_dict, perfect_prior, random_priors[random.randrange(len(random_priors))], llm_consensus_prior)

        # Evaluate Results
        for i in range(len(result)):
            result[i]['shd'] = causal.calculate_shd(ref_dag, result[i]['dag'])
            result[i]['edges'] = causal.precision_recall_f1(ref_dag, result[i]['dag'])

        base, ext = os.path.splitext(data_file)
        result_path = base + '_result.pkl'
        utils.write_pickle(result, results_output_dir + '/' + result_path)

    # Evaluate Causal Discovery
    result_files_list = os.listdir(results_output_dir)
    results = {}
    for file in result_files_list:
        if "_result.pkl" in file:
            results[file] = utils.read_pickle(results_output_dir + '/' + file)
    
    print("Writing report...")
    shd_summary, metrics_summary = causal.generate_discovery_report(results)
    utils.write_json(shd_summary, exp_dir + '/shd_summary.json')
    utils.write_json(metrics_summary, exp_dir + '/metrics_summary.json')
    


if __name__ == "__main__":

    print("Starting Causal Discovery pipeline...")

    for leg in Config.SUPPLY_CHAIN_LEGS:
        run(leg)

    print("Done.")
    

    


