# Async Utils

[![PyPI version](https://img.shields.io/pypi/v/jb_async_utils.svg?maxAge=3600)](https://pypi.org/project/jb_async_utils)
[![PyPI license](https://img.shields.io/pypi/l/jb_async_utils.svg?maxAge=3600)](https://github.com/jacobvsdanniel/async_utils/blob/main/LICENSE)

*Utilities for running asynchronous IO tasks.*

Authors
- Li Peng-Hsuan (李朋軒) @ ailabs.tw (jacobvsdanniel [at] gmail.com)

Features
- A framework of modularized, customizable data definition, quota management, and task runner
- A bare-bones math application for demo purposes
- OpenAI and Deep Infra applications for high-throughput, real-time, batched requests

## Installation

```
python -m pip install jb_async_utils
```

## Usage

For more detailed usage, see *examples*

```python
import json
import asyncio
import tiktoken
from openai import AsyncOpenAI
from async_utils import (
    process_batch_data, OpenAITaskDatum, OpenAIQuotaManager, openai_task_runner
)

input_file = "tmp_in.jsonl"
output_file = "tmp_out.jsonl"

api_key = input("API key: ")
OpenAITaskDatum.client = AsyncOpenAI(api_key=api_key)
OpenAITaskDatum.tokenizer = tiktoken.encoding_for_model("gpt-5")
quota_manager = OpenAIQuotaManager(60, 500)

with open(input_file, "w", encoding="utf8") as f:
    for i in range(10):
        datum = {
            "text_in": f"Reverse the word order of the sentence: I have {i + 1} fish.",
            "model": "gpt-5.2",
            "choices": 1,
        }
        json.dump(datum, f, ensure_ascii=False)
        f.write("\n")

asyncio.run(process_batch_data(
    input_file, output_file, OpenAITaskDatum, openai_task_runner, quota_manager,
    max_task_runs=3, start_id=None, end_id=None, ignore_and_rewrite_output_file=True,
    sleep_interval=0.001,
))
```
