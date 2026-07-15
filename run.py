import argparse
import time
import random
import pandas as pd

from common import utils
from config import Config
import priors
import causal
import os
import risk_attribution

def parse_args():
    p = argparse.ArgumentParser(description="Run the causal discovery pipeline.")
    p.add_argument("--leg", type=str, help=f'{",".join(Config.SUPPLY_CHAIN_LEGS)}')
    return p.parse_args()

def flatten_runs(results_dir):
    result_files_list = os.listdir(results_dir)
    all_runs = []
    for file in result_files_list:
        if "_result.json" in file:
            result = utils.read_json(results_dir + '/' + file)
            for run in result["runs"]:
                lam = (run.get("params") or {}).get("lambda")
                all_runs.append({
                    "name": result["name"], 
                    "seed": result["seed"],
                    "n_rows": result["num_rows"],
                    "lambda": lam,
                    "algo": run["algo"],
                    "type": run["type"],
                    "prior_name": run.get("prior_name"),
                    "shd": run['shd'],
                    "precision": run['precision'],
                    "recall": run['recall'],
                    "f1": run['f1'],
                    "dag_json": run["dag_json"],
                    "attribution_accuracy": run["attribution_accuracy"]
                })
    return all_runs

def run(leg):
    prev = time.time()
    print("="*50)
    print(f"Running Leg : {leg}")

    exp_dir = f'{Config.EXPERIMENTS_DIR}/{leg}'

    # Load Data Dictionary
    data_dict_filepath = f'{exp_dir}/metadata/data_dictionary.json'
    data_dict = utils.read_json(data_dict_filepath)

    # Load Ref DAG pos
    pos_filepath =  f'{exp_dir}/metadata/ref_dag_pos.pkl'
    pos = utils.read_pickle(pos_filepath)

    # Load Ref DAG
    pos_filepath =  f'{exp_dir}/metadata/ref_dag.pkl'
    ref_dag = utils.read_pickle(pos_filepath)

    # Fetch Outcome variable and Causes
    outcome = Config.DELIVERY_RISK[leg]["outcome"]
    causes = Config.DELIVERY_RISK[leg]["causes"]
    scm = utils.get_scm(leg)

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
    print("Writing prior summary..")
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
        continue
        data_filename = data_file['name']
        print(f"\nProcessing {data_filename} .. ")
        df = pd.read_csv(data_dir + '/' + data_filename)

        # Changing random prior for each dataset
        all_discovery_priors['random'] = all_priors['random'][random.randrange(len(all_priors['random']))]

        runs = causal.run_causal_discovery_pipeline(
                                                        df, 
                                                        data_dict, 
                                                        all_discovery_priors)

        # Evaluate individual runs
        for i in range(len(runs)):
            runs[i] = runs[i] | causal.evaluate_graph(ref_dag, runs[i]['dag'])
            runs[i]['dag_json'] = utils.convert_graph_to_json(runs[i]['dag'])

            print(f"Evaluating Attribution ..  {runs[i]['algo']} | {runs[i]['type']}.. ")
            runs[i]['attribution_accuracy'] = risk_attribution.attribution_accuracy(df, runs[i]['dag'], outcome, causes, scm)

            del runs[i]['dag']

        data_file['runs'] = runs

        base, ext = os.path.splitext(data_filename)
        result_path = base + '_result.json'
        utils.write_json(data_file, results_output_dir + '/' + result_path)

    
    all_runs = flatten_runs(results_output_dir)
    
    print("Writing all runs..")
    utils.write_json(all_runs, exp_dir + '/all_runs.json')

    print("Writing discovery report...")
    discovery_summary = causal.generate_discovery_report(all_runs)
    utils.write_json(discovery_summary, exp_dir + '/discovery_summary.json')

    print("Writing attribution report...")
    attribution_summary = risk_attribution.generate_attribution_report(all_runs, data_dir, outcome)
    utils.write_json(attribution_summary, exp_dir + '/attribution_summary.json')
    print(f"Took {time.time()-prev:.1f} secs\n")

    


if __name__ == "__main__":
    args = parse_args()

    print("Starting Causal Discovery pipeline...")
    if args.leg is not None:
        run(args.leg)
    else:
        for leg in Config.SUPPLY_CHAIN_LEGS:
            run(leg)

    print("Done.")
    

    


