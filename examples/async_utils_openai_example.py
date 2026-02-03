import sys
import json
import asyncio
import logging
import argparse

import tiktoken
from openai import AsyncOpenAI

from async_utils import process_batch_data
from async_utils import OpenAITaskDatum, OpenAIQuotaManager
from async_utils import openai_task_runner, dummy_openai_task_runner

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
    OpenAITaskDatum.client = AsyncOpenAI(api_key=api_key)

    # model
    model = "gpt-3.5-turbo"
    OpenAITaskDatum.tokenizer = tiktoken.encoding_for_model(model)
    choices = 2

    # input data
    with open(arg.input_file, "w", encoding="utf8") as f:
        for i in range(100):
            datum = {
                "text_in": f"Reverse the word order of the sentence: 土地公飼養{i + 1}隻吳郭魚。",
                "model": model,
                "choices": choices,
            }
            json.dump(datum, f)
            f.write("\n")

    # quota
    quota_manager = OpenAIQuotaManager(60, 500)

    # dummy run
    # asyncio.run(process_batch_data(
    #     arg.input_file, arg.output_file, OpenAITaskDatum, dummy_openai_task_runner, quota_manager,
    #     max_task_runs=3, start_id=None, end_id=None, ignore_and_rewrite_output_file=True,
    #     sleep_interval=0.001,
    # ))

    # real run
    asyncio.run(process_batch_data(
        arg.input_file, arg.output_file, OpenAITaskDatum, openai_task_runner, quota_manager,
        max_task_runs=3, start_id=None, end_id=None, ignore_and_rewrite_output_file=True,
        sleep_interval=0.001,
    ))
    return


if __name__ == "__main__":
    main()
    sys.exit()
