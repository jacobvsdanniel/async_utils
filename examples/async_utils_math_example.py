import sys
import json
import asyncio
import logging
import argparse

from async_utils import BasicTaskDatum, math_task_runner, BasicQuotaManager, process_batch_data

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

    # create input data
    with open(arg.input_file, "w", encoding="utf8") as f:
        for i in range(15):
            a = i + 1
            b = i + 2
            datum = {"a": a, "b": b}
            json.dump(datum, f)
            f.write("\n")

    # run
    quota_manager = BasicQuotaManager()
    quota_manager.runs_per_minute = 5

    asyncio.run(process_batch_data(
        arg.input_file, arg.output_file, BasicTaskDatum, math_task_runner, quota_manager,
        max_task_runs=2, start_id=None, end_id=None, ignore_and_rewrite_output_file=True,
        sleep_interval=0.001,
    ))
    return


if __name__ == "__main__":
    main()
    sys.exit()
