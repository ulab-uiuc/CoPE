# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import version, PackageNotFoundError
from packaging import version as vs


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)

# Two engine APIs live behind one rollout. Up to 0.6.3 verl vendors a patched vLLM
# (verl/third_party/vllm/vllm_v_0_6_3) whose LLM takes the FSDP module directly and
# exposes sync/offload_model_weights + init/free_cache_engine. From 0.6.6.post2 on
# vLLM supports SPMD inference, and verl drives the stock LLM through the external
# launcher with sleep()/wake_up() in place of the cache-engine calls. vllm_rollout
# implements both; the agent loop itself is identical either way.
#
# The comparison has to go through packaging.version rather than string ordering:
# '0.10.0' sorts before '0.6.3' as a string, which would silently select the
# customized path on a vLLM new enough to have dropped that API.
if vs.parse(package_version) <= vs.parse('0.6.3'):
    vllm_mode = 'customized'
else:
    vllm_mode = 'spmd'

from .vllm_rollout import vLLMRollout
