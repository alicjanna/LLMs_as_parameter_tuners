from typing import List, Union, Tuple, Dict
from mealpy import Problem, Optimizer, IntegerVar
from mealpy.utils.agent import Agent
from mealpy.utils.termination import Termination
import numpy as np
import time


class TSPProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    @staticmethod
    def calculate_distance(city_a, city_b):
        return np.linalg.norm(city_a - city_b)

    @staticmethod
    def calculate_total_distance(route, city_positions):
        total_distance = 0
        num_cities = len(route)
        for idx in range(num_cities):
            current_city = route[idx]
            next_city = route[(idx + 1) % num_cities]
            total_distance += TSPProblem.calculate_distance(city_positions[current_city], city_positions[next_city])
        return total_distance

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        route = x_decoded["per_var"]
        fitness = self.calculate_total_distance(route, self.data["city_positions"])
        return fitness


class GCProblem(Problem):

    def __init__(self, data, **kwargs):
        self.data = data
        num_vertices = len(data)
        bounds = [IntegerVar(lb=1, ub=num_vertices) for _ in range(num_vertices)]

        super().__init__(minmax="min", bounds=bounds, dimension=num_vertices, **kwargs)

    def obj_func(self, solution, **kwargs):
        conflicts = 0
        max_color = 0
        for vertex, color in enumerate(solution):
            max_color = max(max_color, color)
            for neighbor in self.data[vertex]:
                if color == solution[neighbor]:
                    conflicts += 1
        return conflicts + max_color / len(solution)


class JSSProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["per_var"]
        makespan = np.zeros((self.data["n_jobs"], self.data["n_machines"]))
        for gene in x:
            job_idx = gene // self.data["n_machines"]
            machine_idx = gene % self.data["n_machines"]
            if job_idx == 0 and machine_idx == 0:
                makespan[job_idx][machine_idx] = self.data['job_times'][job_idx][machine_idx]
            elif job_idx == 0:
                makespan[job_idx][machine_idx] = makespan[job_idx][machine_idx - 1] + self.data['job_times'][job_idx][machine_idx]
            elif machine_idx == 0:
                makespan[job_idx][machine_idx] = makespan[job_idx - 1][machine_idx] + self.data['job_times'][job_idx][machine_idx]
            else:
                makespan[job_idx][machine_idx] = max(makespan[job_idx][machine_idx - 1], makespan[job_idx - 1][machine_idx]) + self.data['job_times'][job_idx][machine_idx]
        return np.max(makespan)


class OptimizerPrime(Optimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pops = {}

    def solve(self, problem: Union[Dict, Problem] = None, mode: str = 'single', n_workers: int = None,
              termination: Union[Dict, Termination] = None, starting_solutions: Union[List, np.ndarray, Tuple] = None,
              seed: int = None) -> Agent:

        self.check_problem(problem, seed)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_solutions)
        self.initialization()
        self.after_initialization()

        self.before_main_loop()
        self.pops = {}
        for epoch in range(1, self.epoch + 1):
            time_epoch = time.perf_counter()

            # Evolve method will be called in child class
            self.evolve(epoch)
            #print(self.pop) ## TODO HERE
            # Update global best solution, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_agent(self.pop)
            if self.sort_flag: self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch, time_epoch)
            if self.check_termination("end", None, epoch):
                self.pops[epoch] = self.pop
                break
            if (epoch == max(range(1, self.epoch + 1))) or (epoch == 1):
                self.pops[epoch] = self.pop
        self.track_optimize_process()

        return self.g_best
