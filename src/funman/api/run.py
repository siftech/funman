import argparse
import json
import logging
import os
import random
from contextlib import contextmanager
from time import sleep
from timeit import default_timer
from typing import Dict, Tuple, Union

from funman_demo.parameter_space_plotter import ParameterSpacePlotter
from matplotlib import pyplot as plt

import funman
from funman.api.settings import Settings
from funman.model.generated_models.petrinet import Model as GeneratedPetriNet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.model.model import _wrap_with_internal_model
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

l = logging.getLogger(__name__)

# import matplotlib.pyplot as plt


# from funman_demo.parameter_space_plotter import ParameterSpacePlotter


# RESOURCES = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "../resources"
# )

# out_dir = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "out", "evaluation"
# )


models = {GeneratedPetriNet, GeneratedRegnet}

# AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
# AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "amr-examples")
# AMR_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "amr-examples")

# SKEMA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "skema")
# SKEMA_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "skema")

# MIRA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "mira")


# cases = [
#     # S1 base model
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_base.json"),
#     ),
#     # S1 base model ps for beta
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_ps_beta.json"
#         ),
#     ),
#     # S1 1.ii.1
#     (
#         os.path.join(
#             MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_1_init1.json"
#         ),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_1.json"),
#     ),
#     # S1 2 # has issue with integer overflow due to sympy taylor series
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_2.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_2.json"),
#     ),
#     # S1 3, advanced to t=75, parmsynth to separate (non)compliant
#     # (
#     #     os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_3_t75.json"),
#     #     os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_3_t75_ps.json"),
#     # ),
#     # S3 base for CEIMS
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario3_base.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario3_base.json"),
#     ),
# ]

# speedup_cases = [
#     # baseline: no substitution, no mcts, no query simplification
#     # > 10m
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_baseline.json"
#         ),
#         "Baseline",
#     ),
#     # mcts: no substitution, no query simplification
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_mcts.json"
#         ),
#         "MCTS",
#     ),
#     # mcts, substitution, no query simplification
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_substitution.json"
#         ),
#         "MCTS+Sub+Approx",
#     ),
#     # mcts, substitution, query simplification
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_base.json"),
#         "MCTS+Sub+Approx+Compile",
#     ),
# ]


class GracefulKiller:
    kill_now = False

    def __init__(self):
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        # signal.signal(signal.SIGTERM, self.exit_gracefully)
        pass

    def exit_gracefully(self, *args):
        l.info("Requesting FUNMAN to exit because of kill signal ...")
        self.kill_now = True


class Runner:
    @contextmanager
    def elapsed_timer(self):
        start = default_timer()
        elapser = lambda: default_timer() - start
        try:
            yield elapser
        finally:
            elapser = None

    def run(
        self,
        model,
        request,
        description="",
        case_out_dir=".",
        dump_plot=False,
        parameters_to_plot=None,
        point_plot_config={},
        num_points=None,
    ) -> FunmanResults:
        results = self.run_test_case(
            (model, request, description),
            case_out_dir,
            dump_plot=dump_plot,
            parameters_to_plot=parameters_to_plot,
            point_plot_config=point_plot_config,
            num_points=num_points,
        )
        return results
        # ParameterSpacePlotter(
        #     results.parameter_space,
        #     plot_points=True,
        #     parameters=["beta", "num_steps"],
        # ).plot(show=False)
        # plt.savefig(f"{case_out_dir}/scenario1_base_ps_beta_space.png")

    def run_test_case(
        self,
        case,
        case_out_dir,
        dump_plot=False,
        parameters_to_plot=None,
        point_plot_config={},
        num_points=None,
    ):
        if not os.path.exists(case_out_dir):
            os.mkdir(case_out_dir)

        self.settings = Settings()
        self.settings.data_path = case_out_dir
        self._storage = Storage()
        self._worker = FunmanWorker(self._storage)
        self._storage.start(self.settings.data_path)
        self._worker.start()

        results = self.run_instance(
            case,
            out_dir=case_out_dir,
            dump_plot=dump_plot,
            parameters_to_plot=parameters_to_plot,
            point_plot_config=point_plot_config,
            num_points=num_points,
        )

        self._worker.stop()
        self._storage.stop()

        return results

    def get_model(self, model_file: str):
        for model in models:
            try:
                with open(model_file, "r") as mf:
                    j = json.load(mf)
                    m = _wrap_with_internal_model(model(**j))
                return m
            except Exception as e:
                pass
        raise Exception(f"Could not determine the Model type of {model_file}")

    def run_instance(
        self,
        case: Tuple[str, Union[str, Dict], str],
        out_dir=".",
        dump_plot=False,
        parameters_to_plot=None,
        point_plot_config={},
        num_points=None,
    ):
        killer = GracefulKiller()
        (model_file, request_file, description) = case

        model = self.get_model(model_file)

        try:
            with open(request_file, "r") as rf:
                request = FunmanWorkRequest(**json.load(rf))
        except TypeError as te:
            # request_file may not be a path, could be a dict
            try:
                request = FunmanWorkRequest.model_validate(request_file)
            except Exception as e:
                raise e

        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )

        sleep(2)  # need to sleep until worker has a chance to start working
        outfile = f"{out_dir}/{work_unit.id}.json"
        while not killer.kill_now:
            if self._worker.is_processing_id(work_unit.id):
                l.info(f"Dumping results to {outfile}")
                results = self._worker.get_results(work_unit.id)
                with open(outfile, "w") as f:
                    f.write(results.model_dump_json(by_alias=True))
                if dump_plot:
                    points = results.parameter_space.points()
                    if len(points) > 0:
                        point_plot_filename = (
                            f"{out_dir}/{work_unit.id}_points.png"
                        )
                        l.info(
                            f"Creating plot of point trajectories: {point_plot_filename}"
                        )

                        points_to_plot = (
                            random.choices(
                                points, k=min(len(points), num_points)
                            )
                            if num_points
                            else results.parameter_space.points()
                        )
                        results.plot(
                            points=points_to_plot, **point_plot_config
                        )
                        plt.savefig(point_plot_filename)
                        plt.close()

                    boxes = results.parameter_space.boxes()
                    if len(boxes) > 0:
                        space_plot_filename = (
                            f"{out_dir}/{work_unit.id}_parameter_space.png"
                        )
                        l.info(
                            f"Creating plot of parameter space: {space_plot_filename}"
                        )
                        ParameterSpacePlotter(
                            results.parameter_space,
                            plot_points=False,
                            parameters=parameters_to_plot,
                        ).plot(show=False)
                        plt.savefig(space_plot_filename)
                        plt.close()
                sleep(10)
            else:
                results = self._worker.get_results(work_unit.id)
                break

        if killer.kill_now:
            l.info(
                "Requesting that worker stop because received kill signal ..."
            )
            self._worker.stop()
            self._storage.stop()

        # ParameterSpacePlotter(results.parameter_space, plot_points=True).plot(
        #     show=False
        # )
        # plt.savefig(f"{out_dir}/{model.__module__}.png")
        # plt.close()

        return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"model json file",
    )
    parser.add_argument(
        "request",
        type=str,
        help=f"request json file",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="out",
        help=f"Output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    results = Runner().run(args.model, args.request, case_out_dir=args.outdir)
    print(results.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
