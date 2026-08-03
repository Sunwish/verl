# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from verl.utils.device import get_local_device_index


def test_get_local_device_index_matches_visible_slice():
    visible_devices = "4,5,6,7"

    assert get_local_device_index(4, visible_devices=visible_devices) == 0
    assert get_local_device_index(5, visible_devices=visible_devices) == 1
    assert get_local_device_index("6", visible_devices=visible_devices) == 2
    assert get_local_device_index(7, visible_devices=visible_devices) == 3


def test_get_local_device_index_falls_back_to_modulo_for_physical_ids():
    visible_devices = "4,5,6,7"

    assert get_local_device_index(0, visible_devices=visible_devices) == 0
    assert get_local_device_index(1, visible_devices=visible_devices) == 1
    assert get_local_device_index(8, visible_devices=visible_devices) == 0


def test_get_local_device_index_reads_visible_devices_from_env(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")

    assert get_local_device_index(4) == 0
    assert get_local_device_index(7) == 3


def test_get_local_device_index_handles_invalid_input():
    assert get_local_device_index("invalid", visible_devices="4,5,6,7") == 0
