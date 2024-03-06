#!/usr/bin/env python3

import json
import logging
from pathlib import Path

import requests

l = logging.getLogger(__name__)

QUERIES_ENDPOINT = "/api/queries"


def post_query(
    url: str, model_path: str, request_path: str, timeout: float = None
):
    if request_path is None:
        l.info("Falling back to default request of {}")
        request = {}
        payload = [f'"model": <Contents of {model_path}>', '"request": {}']
    else:
        request = read_to_dict(request_path)
        payload = [
            f'"model": <Contents of {model_path}>',
            f'"request": <Contents of {request_path}>',
        ]
    l.error("{")
    for p in payload:
        l.error(f"    {p}")
    l.error("}")

    model = read_to_dict(model_path)
    endpoint = f"{url.rstrip('/')}{QUERIES_ENDPOINT}"
    payload = {"model": model, "request": request}
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return json.loads(response.content.decode())


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
    parser.add_argument("model", type=str, help="the path to the model json")
    parser.add_argument(
        "-r", "--request", type=str, help="the path to the request json"
    )
    args = parser.parse_args()

    results = post_query(args.url, args.model, args.request)
    l.error(f"Query received work id: {results['id']}")
    l.info(results["id"])
