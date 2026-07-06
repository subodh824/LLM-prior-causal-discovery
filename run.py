import argparse
import time
import pandas as pd

from common import utils
from config import Config
import priors
import causal
import evaluate
import os

def parse_args():
    p = argparse.ArgumentParser(description="Run the causal discovery pipeline.")
    p.add_argument("--data", type=str, help="Data dir ")
    return p.parse_args()


if __name__ == "__main__":

    print("Starting Causal Discovery pipeline...")

    experiment = 'distributor_to_customer'
    print(f"Experiment : {experiment}")

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
    priors.generate_priors(data_dict, prior_output_dir) 

    # Load perfect_prior
    perfect_prior = utils.read_json(prior_output_dir + '/perfect_prior.json')

    # Load llm_consensus_prior
    llm_consensus_prior = utils.read_json(prior_output_dir + '/llm_consensus_prior.json')


    # Load data files list
    data_dir =  f'{exp_dir}/data'
    data_files_list = os.listdir(data_dir)

    # Create results dir
    results_output_dir =  f'{exp_dir}/results'
    utils.create_dir_if_not_exists(results_output_dir)

    for data_file in data_files_list:
        print(f"\n Processing {data_file} .. ")
        df = pd.read_csv(data_dir + '/' + data_file)

        # Load random_prior
        random_prior = utils.read_json(prior_output_dir + '/random/random_prior_1.json')

        result = causal.run_causal_discovery_pipeline(df, data_dict, perfect_prior, random_prior, llm_consensus_prior, results_output_dir)

        # Evaluate Results
        print("Evaluating results....")
        for i in range(len(result)):
            result[i]['shd'] = evaluate.calculate_shd(ref_dag, result[i]['dag'])
            result[i]['edges'] = evaluate.precision_recall_f1(ref_dag, result[i]['dag'])

        for d in result:
            d.pop('dag', None)

        base, ext = os.path.splitext(data_file)
        result_path = base + '_result.json'
        utils.write_json(result, results_output_dir + '/' + result_path)

        
    # Evaluate Priors

    # Evaluate Causal Discovery
     
    


    # print("Writing report...")
    # utils.write_json(results, output_dir + 'report.json')

    print("Done.")


