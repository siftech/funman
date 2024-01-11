
import itertools
from funman_demo.generators.advection import main as advection_main
from funman_demo.generators.halfar import main as halfar_main
from funman_demo.generators.common import Derivative
from sys import argv
import os
import shutil

models = {"advection": advection_main, "halfar": halfar_main}

configuration_options = {
    "derivative": {"flag": "-g", "options": [Derivative.CENTERED, Derivative.FORWARD, Derivative.BACKWARD]},
    "dimensions": {"flag": "-d", "options": [1, 2, 3]},
    "boundary": {"flag": "-b", "options": [0.0, 0.01, 0.1, 0.2]},
    "points": {"flag": "-p", "options": [3, 5, 10]},
}

def outfile_for_option(path, model, options):
    filename = f"{model}_" + "_".join([str(o).replace("-", "n") for o in options]) + ".json"
    modelpath = os.path.join(path, model)
    filepath = os.path.join(modelpath, filename)
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    return filepath

def options():
    return itertools.product(*[v['options'] for v in configuration_options.values()])

def main(filepath="."):
    for model in models:
        print(f"model: {model}")
        for option in options():
            filename = outfile_for_option(".", model, option)
            opts = list(zip([v['flag'] for v in configuration_options.values()], option))
            opts = [str(v1) for v in opts for v1 in v] + ["-o", filename]
            print(f"filename: {filename}, option: {opts}")

            models[model](opts)

if __name__ == "__main__":
    if len(argv) > 1:
        main(filepath=argv[1])
    else:
        main()