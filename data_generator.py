import argparse
import time
import os
from config import Config

from common import utils
import priors

def parse_args():
    p = argparse.ArgumentParser(description="Run the data generation script.")
    p.add_argument("--create-dag", type=str, help="")
    p.add_argument("--leg", type=str, help="distributor_to_customer,manufacturer_to_distributor,supplier_to_manufacturer")
    p.add_argument("--num-rows", type=int, default=500, help="Number of rows")
    return p.parse_args()

def generate_metadata(scm, output_dir):
    scm.build_ref_graph(output_dir)
    scm.build_data_dictionary(output_dir)
    return


def generate_data(scm, num_rows, seed):
    df = scm.generate(num_rows, seed)
    return df


if __name__ == "__main__":
    args = parse_args()

    num_rows = [250, 1000]
    seeds = [3, 42, 5]

    print("Generating synthetic data ...")

    t0 = time.time()

    print(f"\nCreate data dir '{Config.EXPERIMENTS_DIR}' if not exists...\n")
    utils.create_dir_if_not_exists(Config.EXPERIMENTS_DIR)

    for leg in Config.SUPPLY_CHAIN_LEGS:
        print(f"Generating data for {leg}..\n")

        scm = utils.get_scm(leg)

        output_dir = Config.EXPERIMENTS_DIR + '/' + leg
        utils.create_dir_if_not_exists(output_dir)

        print("Generating metadata ..")
        metadata_output_dir = output_dir + '/metadata'
        utils.create_dir_if_not_exists(metadata_output_dir)

        generate_metadata(scm, metadata_output_dir)

        data_output_dir = output_dir + '/data'
        utils.create_dir_if_not_exists(data_output_dir)
        data_files_list = []
        for seed in seeds:
            for n in num_rows:
                print(f"Generating data for Leg: {leg} | Seed : {seed} | Size : {n}")
                df = generate_data(scm, n, seed)

                data_file_name = f'syn_data_{seed}_{n}.csv'
                df.to_csv(os.path.join(data_output_dir, data_file_name), index=False)
                data_files_list.append({
                    'name': data_file_name,
                    'seed': seed,
                    'num_rows': n
                })

        utils.write_json(data_files_list, metadata_output_dir + '/data_files_list.json')
    
    print(f"\nDone in {(time.time() - t0):.1f} secs.\n")