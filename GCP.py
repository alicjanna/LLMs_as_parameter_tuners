import pandas as pd
import glob
from collections import defaultdict
from utils import GCProblem, OptimizerPrime
from mealpy import PermutationVar, GA, SA, TS, PSO
import pickle
import mock
import sys
import numpy as np

np.random.seed(2137)


def read_instance(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        parsed = []
        for line in lines[1:]:
            if line.strip():
                numbers = line.strip().split()[1:]
                numbers = [int(x) for x in numbers]
                parsed.append(numbers)

        graph = defaultdict(list)
        for node, connected_node in parsed:
            graph[node].append(connected_node)
        graph = dict(graph)

        zero_based_graph = {k - 1: [v - 1 for v in values] for k, values in graph.items()}

        for node, neighbors in list(zero_based_graph.items()):
            for neighbor in neighbors:
                if node not in zero_based_graph.get(neighbor, []):
                    # Add the node to the neighbor's list if it's not already there
                    if neighbor not in zero_based_graph:
                        zero_based_graph[neighbor] = [node]
                    else:
                        zero_based_graph[neighbor].append(node)
    return zero_based_graph


# instances
instances_path = 'instances_04_GCP'
instances_files = glob.glob('{}/*'.format(instances_path))
instances_names = [x.split('\\')[-1] for x in instances_files]
instances_data = [read_instance(x) for x in instances_files]

instances = dict(zip(instances_names, instances_data))

# global
PROBLEM = 'GCP'
MHA = 'SA'
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

    for run_id in range(30):

        if algorithm == 'GA':

            problem = GCProblem(data=instances[problemix], log_to='file',
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

            problem = GCProblem(data=instances[problemix], log_to='file',
                                log_file='{}/{}_{}_{}.log'.format(OUTPUT_DIR, algorithm, run_type, idx),
                                name='{}_{}_{}_{}_{}'.format(algorithm, source, run_type, run_id, problemix))

            pacz = mock.patch.object(SA.OriginalSA, '__bases__', (OptimizerPrime,))
            with pacz:
                pacz.is_local = True
                model = SA.OriginalSA(epoch=row['epoch'],
                                      pop_size=row['pop_size'],
                                      temp_init=row['temp_init'],
                                      step_size=row['step_size'])
                model.solve(problem)

        elif algorithm == 'PSO':

            problem = GCProblem(data=instances[problemix], log_to='file',
                                log_file='{}/{}_{}_{}.log'.format(OUTPUT_DIR, algorithm, run_type, idx),
                                name='{}_{}_{}_{}_{}'.format(algorithm, source, run_type, run_id, problemix))

            pacz = mock.patch.object(PSO.OriginalPSO, '__bases__', (OptimizerPrime,))
            with pacz:
                pacz.is_local = True
                model = PSO.OriginalPSO(pop_size=row['pop_size'],
                                        c1=row['c1'],
                                        c2=row['c2'],
                                        w=row['w'],
                                        epoch=row['epoch'])

                model.solve(problem)

                with open("{}/population_{}_{}_{}_{}_{}.pkl".format(OUTPUT_DIR, algorithm, source, run_type, run_id,
                                                                    problemix),
                          "wb") as outfile:
                    pickle.dump(model.pops, outfile)
