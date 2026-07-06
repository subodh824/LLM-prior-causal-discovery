import argparse
import time
import os
from config import Config

from dags import distributor_to_customer 
from dags import manufacturer_to_distributor
from dags import supplier_to_manufacturer

from common import utils
import priors

def parse_args():
    p = argparse.ArgumentParser(description="Run the data generation script.")
    p.add_argument("--create-dag", type=str, help="")
    p.add_argument("--leg", type=str, help="distributor_to_customer,manufacturer_to_distributor,supplier_to_manufacturer")
    p.add_argument("--num-rows", type=int, default=500, help="Number of rows")
    return p.parse_args()

def generate_metadata(module, output_dir):
    module.build_ref_graph(os.path.join(output_dir, 'ref_dag.png'))
    return module.build_data_dictionary(os.path.join(output_dir, 'data_dictionary.json'))


def generate_data(module, output_dir, num_rows, seed):
    df = module.generate(num_rows, seed)
    df.to_csv(os.path.join(output_dir, f'syn_data_{seed}_{num_rows}.csv'), index=False)
    return df


if __name__ == "__main__":
    args = parse_args()

    num_rows = [250, 1000, 2000]
    seeds = [42]

    leg_to_module = {
        Config.DISTRIBUTION_TO_CUSTOMER: distributor_to_customer,
        Config.MANUFACTURER_TO_DISTRIBUTOR: manufacturer_to_distributor,
        Config.SUPPLIER_TO_MANUFACTURER: supplier_to_manufacturer,
    }

    print("Generating synthetic data ...")

    t0 = time.time()

    print(f"\nCreate data dir '{Config.EXPERIMENTS_DIR}' if not exists...\n")
    utils.create_dir_if_not_exists(Config.EXPERIMENTS_DIR)

    for leg in Config.SUPPLY_CHAIN_LEGS:
        print(f"Generating data for {leg}..\n")

        output_dir = Config.EXPERIMENTS_DIR + '/' + leg
        utils.create_dir_if_not_exists(output_dir)

        print("Generating metadata ..")
        metadata_output_dir = output_dir + '/metadata'
        utils.create_dir_if_not_exists(metadata_output_dir)
        
        generate_metadata(leg_to_module[leg], metadata_output_dir)

        # print("Generating priors ..")
        # prior_output_dir = output_dir + '/priors'
        # utils.create_dir_if_not_exists(prior_output_dir)
        # priors.generate_priors(data_dict, prior_output_dir)

        data_output_dir = output_dir + '/data'
        utils.create_dir_if_not_exists(data_output_dir)
        for seed in seeds:
            for n in num_rows:
                print(f"Generating data for Leg: {leg} | Seed : {seed} | Size : {n}")
                generate_data(leg_to_module[leg], data_output_dir, n, seed)
    
    print(f"\nDone in {(time.time() - t0):.1f} secs.\n")