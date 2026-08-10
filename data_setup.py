import argparse
import time
import os
from config import Config

from common import utils
import priors


def generate_metadata(scm, output_dir):
    scm.build_ref_graph(output_dir)
    scm.build_data_dictionary(output_dir)
    scm.build_downstream_config(output_dir)
    return

def generate_data(scm, num_rows, seed):
    return scm.generate(num_rows, seed)

def generate_synthetic_data(leg, seeds, num_rows):
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


def process_real_data(leg):
    print(f"Processing data for {leg}..\n")

    scm = utils.get_scm(leg)

    output_dir = Config.EXPERIMENTS_DIR + '/' + leg
    utils.create_dir_if_not_exists(output_dir)

    scm.process(output_dir)



if __name__ == "__main__":
    args = utils.parse_args()

    num_rows = [250, 1000, 5000]
    seeds = [3, 42, 5, 7, 11, 17, 23, 29, 41, 53, 67, 79, 89, 97, 101]

    t0 = time.time()

    print(f"\nCreate data dir '{Config.EXPERIMENTS_DIR}' if not exists...\n")
    utils.create_dir_if_not_exists(Config.EXPERIMENTS_DIR)

    for leg in Config.SUPPLY_CHAIN_LEGS:
        if leg == 'dataco':
            process_real_data(leg)
        else:
            generate_synthetic_data(leg, seeds, num_rows)
    
    print(f"\nDone in {(time.time() - t0):.1f} secs.\n")