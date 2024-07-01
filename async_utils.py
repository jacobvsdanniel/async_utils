import os
import json
import time
import heapq
import random
import asyncio
import logging
import datetime
from collections import deque

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

"""
basic
"""


class BasicTaskDatum:
    def __init__(
            self, task_id, data,
    ):
        self.task_id = task_id
        self.data = data

        self.run_id = 0
        self.start_time = 0
        self.end_time = 0
        return

    def get_log_string(self):
        return f"task#{self.task_id} run#{self.run_id}"

    def get_json_obj(self):
        start_time = datetime.datetime.fromtimestamp(self.start_time).isoformat()
        end_time = datetime.datetime.fromtimestamp(self.end_time).isoformat()

        json_obj = {
            "task_id": self.task_id,
            "data": self.data,
            "start_time": start_time,
            "end_time": end_time,
        }
        return json_obj


class BasicQuotaManager:
    def __init__(self):
        self.runs_per_minute = 5
        return

    def has_enough_quota(self, init_task_datum):
        return self.runs_per_minute > 0

    def reclaim_quota(self, done_task_datum_queue):
        while done_task_datum_queue:
            end_time, _done_task_datum_queue_id, _done_task_datum = done_task_datum_queue[0]
            if end_time >= time.time() - 60:
                break
            heapq.heappop(done_task_datum_queue)
            self.runs_per_minute += 1
        return

    def deduct_quota(self, init_task_datum):
        self.runs_per_minute -= 1
        return


async def process_batch_data(
        input_file, output_file, task_datum_class, task_runner, quota_manager,
        max_task_runs=1, start_id=None, end_id=None, ignore_and_rewrite_output_file=False,
        sleep_interval=0.001,
):
    # input file
    fr = open(input_file, "r", encoding="utf8")
    input_task_id = 0
    no_more_input = False

    # tasks
    todo_task_datum_queue = deque()
    running_task_to_datum = {}
    done_task_datum_queue = []
    done_task_datum_queue_next_id = 0

    # output file
    completed_task_id_set = set()
    if not ignore_and_rewrite_output_file and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf8") as f:
            for line in f:
                datum = json.loads(line)
                completed_task_id_set.add(datum["task_id"])

    output_mode = "w" if ignore_and_rewrite_output_file else "a"
    fw = open(output_file, output_mode, encoding="utf8")

    # loop
    while True:
        await asyncio.sleep(sleep_interval)

        # step 1: loop through running tasks: check and process their completion
        new_running_task_to_datum = {}

        for running_task, running_task_datum in running_task_to_datum.items():
            if running_task.done():
                try:
                    _ = running_task.result()
                    successful = True
                except:
                    successful = False

                if successful:
                    logger.info(f"[success] {running_task_datum.get_log_string()}")
                    json.dump(running_task_datum.get_json_obj(), fw)
                    fw.write("\n")
                    fw.flush()
                else:
                    running_task_datum.end_time = time.time()
                    if running_task_datum.run_id < max_task_runs:
                        logger.info(f"[error] {running_task_datum.get_log_string()}")
                        todo_task_datum_queue.append(running_task_datum)
                    else:
                        logger.info(f"[error] [quit] {running_task_datum.get_log_string()}")

                heapq.heappush(
                    done_task_datum_queue,
                    (
                        running_task_datum.end_time,
                        done_task_datum_queue_next_id,
                        running_task_datum,
                    ),
                )
                done_task_datum_queue_next_id += 1
            else:
                new_running_task_to_datum[running_task] = running_task_datum

        running_task_to_datum = new_running_task_to_datum

        # step 2: loop through done tasks: reclaim quota
        quota_manager.reclaim_quota(done_task_datum_queue)

        # step 3: read more task datum from input file
        if not todo_task_datum_queue and not no_more_input:
            while True:
                line = fr.readline()
                if not line:
                    no_more_input = True
                    break
                input_task_id += 1
                if start_id is not None and input_task_id < start_id:
                    continue
                if end_id is not None and input_task_id > end_id:
                    continue
                if input_task_id in completed_task_id_set:
                    continue
                line_data = json.loads(line)
                input_task_datum = task_datum_class(input_task_id, line_data)
                todo_task_datum_queue.append(input_task_datum)
                break

        # step 4: end if there is no running tasks and no todo tasks
        if not todo_task_datum_queue:
            if running_task_to_datum:
                continue
            else:
                break

        # step 5: run next task
        if quota_manager.has_enough_quota(todo_task_datum_queue[0]):
            init_task_datum = todo_task_datum_queue.popleft()
            init_task_datum.run_id += 1
            quota_manager.deduct_quota(init_task_datum)
            init_task = asyncio.create_task(task_runner(init_task_datum))
            running_task_to_datum[init_task] = init_task_datum
            logger.info(f"[run] {init_task_datum.get_log_string()}")

    fr.close()
    fw.close()
    logger.info("done")
    return


"""
math
"""


async def math_task_runner(task_datum):
    task_datum.start_time = time.time()
    await asyncio.sleep(10)
    result = task_datum.data["a"] * task_datum.data["b"]
    assert random.random() < 0.5
    task_datum.end_time = time.time()

    task_datum.data["result"] = result
    return task_datum


"""
OpenAI
"""


class OpenAITaskDatum(BasicTaskDatum):
    tokenizer = None
    client = None

    def __init__(self, task_id, data):
        super().__init__(task_id, data)

        try:
            self.data["in_tokens"] = len(self.tokenizer.encode(self.data["text_in"]))
        except AttributeError:
            pass
        self.data["text_out_list"] = []
        return

    def set_out_tokens(self):
        self.data["out_tokens"] = sum(
            len(self.tokenizer.encode(text_out))
            for text_out in self.data["text_out_list"]
        )
        return

    @classmethod
    def set_tokenizer(cls, tokenizer):
        cls.tokenizer = tokenizer
        return

    @classmethod
    def set_client(cls, client):
        cls.client = client
        return


class OpenAIQuotaManager(BasicQuotaManager):
    def __init__(self, rpm, tpm):
        super().__init__()
        self.rpm = rpm
        self.tpm = tpm
        return

    def has_enough_quota(self, init_task_datum):
        return self.rpm > 0 and self.tpm > init_task_datum.data["in_tokens"] * 2

    def reclaim_quota(self, done_task_datum_queue):
        while done_task_datum_queue:
            end_time, _done_task_datum_queue_id, done_task_datum = done_task_datum_queue[0]
            if end_time >= time.time() - 60:
                break
            heapq.heappop(done_task_datum_queue)
            self.rpm += 1
            self.tpm += done_task_datum.data["in_tokens"]
        return

    def deduct_quota(self, init_task_datum):
        self.rpm -= 1
        self.tpm -= init_task_datum.data["in_tokens"]
        return


async def openai_task_runner(task_datum):
    task_datum.start_time = time.time()
    completion = await task_datum.client.chat.completions.create(
        model=task_datum.data["model"],
        n=task_datum.data["choices"],
        messages=[
            {"role": "user", "content": task_datum.data["text_in"]},
        ]
    )
    task_datum.end_time = time.time()

    task_datum.data["text_out_list"] = [
        choice.message.content
        for choice in completion.choices
    ]
    task_datum.set_out_tokens()

    return task_datum


async def dummy_openai_task_runner(task_datum):
    task_datum.start_time = time.time()
    await asyncio.sleep(10)
    assert random.random() < 0.5
    task_datum.end_time = time.time()

    task_datum.data["text_out_list"] = [
        f"output-{i + 1}"
        for i in range(task_datum.data["choices"])
    ]
    task_datum.set_out_tokens()

    return task_datum

