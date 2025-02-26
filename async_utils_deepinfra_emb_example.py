import sys
import json
import asyncio
import logging
import argparse

from openai import AsyncOpenAI

from async_utils import process_batch_data
from async_utils import DeepInfraEmbTaskDatum, DeepInfraQuotaManager
from async_utils import deepinfra_emb_task_runner

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def show_result(arg):
    import struct
    import numpy as np

    emb_dimension = 1024
    vector_size = emb_dimension * 8  # a float is 8 bytes

    logger.info(f"reading embedding bytes...")
    with open(arg.emb_file, "rb") as f:
        emb_bytes = f.read()
    total_bytes = len(emb_bytes)
    expected_vectors = total_bytes / vector_size
    logger.info(f"read {total_bytes:,} bytes, expected {expected_vectors:,} vectors")

    logger.info("collecting embedding...")
    bytes_i = 0
    data = []
    with open(arg.output_file, "r", encoding="utf8") as f:
        for line in f:
            datum = json.loads(line)
            text = datum["data"]["text_in"]
            bytes_j = bytes_i + vector_size
            vector = [v[0] for v in struct.iter_unpack("d", emb_bytes[bytes_i:bytes_j])]
            vector = np.array(vector, dtype=np.float64)
            bytes_i = bytes_j
            data.append((text, vector))
    logger.info(f"collected {len(data):,} text vectors")

    # show text
    for di, (text, _vector) in enumerate(data):
        di += 1
        logger.info(f"#{di}: {text}")

    # show similarity
    for di, (ti, vi) in enumerate(data):
        di += 1
        for dj, (tj, vj) in enumerate(data):
            dj += 1
            if di >= dj:
                continue
            similarity = np.dot(vi, vj)
            logger.info("")
            logger.info(f"#{di}: {ti}")
            logger.info(f"#{dj}: {tj}")
            logger.info(f"sim({di},{dj})={similarity:.0%}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--emb_file", type=str)
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[arg.{key}] {value}")

    # show result
    # show_result(arg)
    # return

    # client
    api_key = input("API key: ")
    logger.info("received API key")
    DeepInfraEmbTaskDatum.client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    DeepInfraEmbTaskDatum.bytes_file = open(arg.emb_file, "wb")

    # model
    model = "BAAI/bge-m3"

    # input data
    text_list = [
        "I have a dream.",
        "I had a dream.",
        "Keyboard is useful",
        "Dogs bark.",
        "Dogs can bark.",
    ]

    with open(arg.input_file, "w", encoding="utf8") as f:
        for text in text_list:
            datum = {
                "text_in": text,
                "model": model,
            }
            json.dump(datum, f)
            f.write("\n")

    # quota
    quota_manager = DeepInfraQuotaManager(200)

    # run
    asyncio.run(process_batch_data(
        arg.input_file, arg.output_file, DeepInfraEmbTaskDatum, deepinfra_emb_task_runner, quota_manager,
        max_task_runs=3, start_id=None, end_id=None, ignore_and_rewrite_output_file=True,
        sleep_interval=0.001,
    ))
    return


if __name__ == "__main__":
    main()
    sys.exit()
