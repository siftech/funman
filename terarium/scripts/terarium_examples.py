#!/usr/bin/env python3

import json
import logging
import sys
from pathlib import Path
from time import sleep

from get_status import get_status, make_status_endpoint
from post_query import QUERIES_ENDPOINT, post_query

l = logging.getLogger(__name__)


def read_to_dict(path: str):
    fpath = Path(path).resolve()
    if not fpath.exists():
        raise FileNotFoundError(f"{path} not found")
    if not fpath.is_file():
        raise Exception(f"{path} is not a file")
    return json.loads(fpath.read_bytes())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="POST query to funman")
    parser.add_argument(
        "url", type=str, help="The base URL of the funman server"
    )
    parser.add_argument(
        "tests", type=str, help="the path to the tests definition"
    )
    parser.add_argument(
        "-r", "--run", type=str, help="the name of the example to run"
    )
    args = parser.parse_args()

    prefix = Path(args.tests).parent.resolve()
    delim = f'{"#" * 80}\n{"#" * 80}'

    tests = read_to_dict(args.tests)["tests"]
    examples = {test["name"]: test for test in tests}
    names = list(examples.keys())

    if args.run is None:
        l.info("Available examples:")
        l.info("-  " + "\n-  ".join(names))
        sys.exit()
    if args.run not in examples:
        l.info(f"'{args.run}' is not a valid example!")
        l.info("Available examples:")
        l.info("-  " + "\n-  ".join(names))
        sys.exit()

    example = examples[args.run]
    l.info(f"Running example '{args.run}':\n{json.dumps(example, indent=2)}")

    model_path = prefix / example["model-path"]
    request_path = None
    if example["request-path"] is not None:
        request_path = prefix / example["request-path"]
    l.info(delim)

    l.info(
        f"Making POST request to {args.url}{QUERIES_ENDPOINT} with contents:"
    )
    results = post_query(args.url, model_path, request_path)
    l.info(delim)

    l.info("Response for query:")
    results["model"] = "Removed for brevity"
    results["request"] = "Removed for brevity"
    l.info(json.dumps(results, indent=2))
    l.info(delim)

    work_id = results["id"]
    l.info(f"Work Id is '{work_id}'")
    l.info("Waiting for 5 seconds...")
    sleep(5)
    l.info(delim)

    l.info(f"Making GET request to {args.url}{make_status_endpoint(work_id)}:")
    results = get_status(args.url, work_id)
    results["model"] = "Removed for brevity"
    results["request"] = "Removed for brevity"
    l.info(json.dumps(results, indent=2))
    l.info(delim)

    l.info("The resulting ParameterSpace is:")
    l.info(json.dumps(results["parameter_space"], indent=2))
