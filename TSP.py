import pandas as pd
import numpy as np
import glob
from utils import TSPProblem, OptimizerPrime
from mealpy import PermutationVar, GA, SA, ACOR
import pickle
import mock
import sys

np.random.seed(911)


def read_instance(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        results = []
        for line in lines:
            if line.strip():
                city_positions = [[float(num) for num in line.split()] for line in line.strip().split('\n')]
                results.append(city_positions)
        data_dict = {"city_positions": np.array(results)[:, 0, 1:3], "num_cities": len(np.array(results))}
    return data_dict


# instances
instances_path = 'instances_02_TSP'
instances_files = glob.glob('{}/*'.format(instances_path))
instances_names = [x.split('\\')[-1] for x in instances_files]
instances_data = [read_instance(x) for x in instances_files]

instances = dict(zip(instances_names, instances_data))

# global
PROBLEM = 'TSP'
MHA = 'GA'
OUTPUT_DIR = '2025/results/{}-{}'.format(PROBLEM, MHA)
PARAMS_DIR = '2025/parameters_main.xlsx'
RUN_TYPE = 'Initial'  # or 'Feedback'

# params
params_book = pd.read_excel(PARAMS_DIR, sheet_name=PROBLEM + '-' + MHA)
params_book = params_book.loc[params_book['run'] == RUN_TYPE]


for index, row in params_book.iterrows():
    print('row: {}'.format(index))

    algorithm = row['MHA']
    run_type = row['run']
    source = row['llm']
    problemix = row['instance']
    idx = ''

    bounds = PermutationVar(valid_set=list(range(0, instances[problemix]['num_cities'])), name="per_var")

    for run_id in range(30):

        if algorithm == 'GA':

            problem = TSPProblem(bounds=bounds, minmax="min", data=instances[problemix], log_to='file',
                                 log_file='{}/{}_{}_{}.log'.format(OUTPUT_DIR, algorithm, run_type, idx),
                                 name='{}_{}_{}_{}_{}'.format(algorithm, source, run_type, run_id, problemix))

            pacz = mock.patch.object(GA.BaseGA, '__bases__', (OptimizerPrime,))
            with pacz:
                pacz.is_local = True
                model = GA.BaseGA(pop_size=row['pop_size'],
                                  selection=row['selection'],
                                  pc=row['pc'],
                                  crossover=row['crossover'],
                                  pm=row['pm'],
                                  mutation_multipoints=row['mutation_multipoints'],
                                  mutation=row['mutation'],
                                  epoch=row['epoch'])

                model.solve(problem)

                with open("{}/population_{}_{}_{}_{}_{}.pkl".format(OUTPUT_DIR, algorithm, source, run_type, run_id,
                                                                    problemix),
                          "wb") as outfile:
                    pickle.dump(model.pops, outfile)

        elif algorithm == 'SA':

            problem = TSPProblem(bounds=bounds, minmax="min", data=instances[problemix], log_to='file',
                                 log_file='{}/{}_{}_{}.log'.format(OUTPUT_DIR, algorithm, run_type, idx),
                                 name='{}_{}_{}_{}_{}'.format(algorithm, source, run_type, run_id, problemix))

            pacz = mock.patch.object(SA.OriginalSA, '__bases__', (OptimizerPrime,))
            with pacz:
                pacz.is_local = True
                model = SA.OriginalSA(pop_size=row['pop_size'],
                                      temp_init=row['temp_init'],
                                      step_size=row['step_size'],
                                      epoch=row['epoch'])

                model.solve(problem)

        elif algorithm == 'ACO':

            problem = TSPProblem(bounds=bounds, minmax="min", data=instances[problemix], log_to='file',
                                 log_file='{}/{}_{}_{}.log'.format(OUTPUT_DIR, algorithm, run_type, idx),
                                 name='{}_{}_{}_{}_{}'.format(algorithm, source, run_type, run_id, problemix))

            pacz = mock.patch.object(ACOR.OriginalACOR, '__bases__', (OptimizerPrime,))
            with pacz:
                pacz.is_local = True
                model = ACOR.OriginalACOR(pop_size=row['pop_size'],
                                          epoch=row['epoch'],
                                          sample_count=row['sample_count'],
                                          intent_factor=row['intent_factor'],
                                          zeta=row['zeta'])

                model.solve(problem)

                with open("{}/population_{}_{}_{}_{}_{}.pkl".format(OUTPUT_DIR, algorithm, source, run_type, run_id,
                                                                    problemix),
                          "wb") as outfile:
                    pickle.dump(model.pops, outfile)
