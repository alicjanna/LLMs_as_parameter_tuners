import pandas as pd
import glob
from utils import JSSProblem, OptimizerPrime
from mealpy import PermutationVar, GA, PSO, ACOR
import pickle
import mock
import sys

np.random.seed(42)


def read_instance(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        data_dict = {'job_times': []}
        jobs, machines = map(float, lines[0].strip().split())
        data_dict['n_jobs'] = int(jobs)
        data_dict['n_machines'] = int(machines)
        data_dict['machine_times'] = [[] for _ in range(int(machines))]
        for line in lines[1:]:
            if line.strip():
                numbers = line.strip().split()
                result = []
                for i in range(0, len(numbers), 2):
                    position = int(numbers[i])  # Get the position
                    value = int(numbers[i + 1])  # Get the value
                    result.append((position, value))
                sorted_data = sorted(result, key=lambda x: x[0])
                result = [x[1] for x in sorted_data]
                data_dict['job_times'].append(result)
    return data_dict


# instances
instances_path = 'instances_03_JSSP'
instances_files = glob.glob('{}/*'.format(instances_path))
instances_names = [x.split('\\')[-1] for x in instances_files]
instances_data = [read_instance(x) for x in instances_files]

instances = dict(zip(instances_names, instances_data))

# global
PROBLEM = 'JSSP'
MHA = 'PSO'
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

    bounds = PermutationVar(
        valid_set=list(range(0, int(instances[problemix]['n_jobs'] * instances[problemix]['n_machines']))),
        name="per_var")

    for run_id in range(30):

        if algorithm == 'GA':

            problem = JSSProblem(bounds=bounds, minmax="min", data=instances[problemix], log_to='file',
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

        elif algorithm == 'PSO':

            problem = JSSProblem(bounds=bounds, minmax="min", data=instances[problemix], log_to='file',
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

        elif algorithm == 'ACO':

            problem = JSSProblem(bounds=bounds, minmax="min", data=instances[problemix], log_to='file',
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
