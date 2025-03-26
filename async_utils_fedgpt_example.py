import sys
import json
import asyncio
import logging
import argparse

from async_utils import process_batch_data
from async_utils import FedGPTTaskDatum, FedGPTQuotaManager
from async_utils import fedgpt_task_runner

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[arg.{key}] {value}")

    # client
    api_key = input("API key: ")
    logger.info("received API key")
    FedGPTTaskDatum.api_key = api_key
    FedGPTTaskDatum.api_url = "https://10.7.240.11/api/chat/v1/chat"

    # model
    model = "fedgpt-medium"

    # input data
    with open(arg.input_file, "w", encoding="utf8") as f:
        for i in range(10):
            datum = {
                "text_in": f"Reverse the word order of the sentence: I have {i + 1} fish.",
                "model": model,
            }
            json.dump(datum, f)
            f.write("\n")

    # quota
    quota_manager = FedGPTQuotaManager(max_concurrent_requests=10)

    # run
    asyncio.run(process_batch_data(
        arg.input_file, arg.output_file, FedGPTTaskDatum, fedgpt_task_runner, quota_manager,
        max_task_runs=3, start_id=None, end_id=None, ignore_and_rewrite_output_file=True,
        sleep_interval=0.001,
    ))
    return


if __name__ == "__main__":
    main()
    sys.exit()
