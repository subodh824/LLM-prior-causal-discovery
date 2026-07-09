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
    module.build_ref_graph(output_dir)
    module.build_data_dictionary(output_dir)
    return


def generate_data(module, num_rows, seed):
    df = module.generate(num_rows, seed)
    return df


if __name__ == "__main__":
    args = parse_args()

    num_rows = [250, 1000, 2000]
    seeds = [3, 42, 65, 76]

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

        data_output_dir = output_dir + '/data'
        utils.create_dir_if_not_exists(data_output_dir)
        data_files_list = []
        for seed in seeds:
            for n in num_rows:
                print(f"Generating data for Leg: {leg} | Seed : {seed} | Size : {n}")
                df = generate_data(leg_to_module[leg], n, seed)

                data_file_name = f'syn_data_{seed}_{n}.csv'
                df.to_csv(os.path.join(data_output_dir, data_file_name), index=False)
                data_files_list.append({
                    'name': data_file_name,
                    'seed': seed,
                    'num_rows': n
                })

        utils.write_json(data_files_list, metadata_output_dir + '/data_files_list.json')
    
    print(f"\nDone in {(time.time() - t0):.1f} secs.\n")