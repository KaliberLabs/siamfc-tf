from collections import namedtuple
import json
import os

parameters_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../parameters")


def load_parameters(file_name):
    with open(os.path.join(parameters_dir, file_name)) as json_file:
        return json.load(json_file)


def parse_arguments(in_hp=None, in_evaluation=None, in_run=None):
    """
    Return hyperparameter, evaluation, run, env, and design config named tuples
    """
    in_hp = {}         if in_hp         is None else in_hp 
    in_evaluation = {} if in_evaluation is None else in_evaluation
    in_run = {}        if in_run        is None else in_run

    hp = load_parameters("hyperparams.json")
    evaluation = load_parameters("evaluation.json")
    run = load_parameters("run.json")
    env = load_parameters("environment.json")
    design = load_parameters("design.json")

    hp.update(in_hp)
    evaluation.update(in_evaluation)
    run.update(in_run)

    # Wrap dicts into named tuples
    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
