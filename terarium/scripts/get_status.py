#!/usr/bin/env python3

import json
import logging

import requests
from post_query import QUERIES_ENDPOINT

l = logging.getLogger(__name__)


def make_status_endpoint(work_id):
    return f"{QUERIES_ENDPOINT}/{work_id}"


def get_status(url: str, uuid: str, timeout: float = None):
    endpoint = f"{url.rstrip('/')}{make_status_endpoint(uuid)}"
    response = requests.get(endpoint, timeout=timeout)
    response.raise_for_status()
    return json.loads(response.content.decode())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GET query status from funman"
    )
    parser.add_argument(
        "url", type=str, help="The base URL of the funman server"
    )
    parser.add_argument("uuid", type=str, help="the uuid of the request")
    args = parser.parse_args()

    l.info(f"Getting status for query: {args.uuid}")
    results = get_status(args.url, args.uuid)

    results["model"] = "Removed for brevity"
    results["request"] = "Removed for brevity"
    l.info(json.dumps(results, indent=2))
