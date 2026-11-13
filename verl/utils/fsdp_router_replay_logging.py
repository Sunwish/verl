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
"""Persist R3 WARNING evidence independently of worker logging configuration."""

import logging
import os
import socket
import sys
import threading
from pathlib import Path

_lock = threading.RLock()
_formatter = logging.Formatter("%(levelname)s:%(asctime)s:pid=%(process)s:%(message)s")


def r3_warning(message, *args, exc_info=False):
    """Write each process's evidence before forwarding a genuine WARNING record.

    Worker libraries can change root levels, disable loggers, or suppress global
    logging. Auditable R3 records must survive those changes. File errors are
    intentionally propagated instead of silently losing validation evidence.
    A separate file per host/PID avoids interleaved multi-process writes; each
    append closes the file, so abnormal process teardown cannot strand buffers.
    """
    if exc_info is True:
        exc_info = sys.exc_info()
    logger = logging.getLogger("verl.r3")
    record = logger.makeRecord(logger.name, logging.WARNING, __file__, 0, message, args, exc_info or None)
    with _lock:
        directory = os.environ.get("VERL_R3_LOG_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"r3-{socket.gethostname()}-{os.getpid()}.log").open("a", encoding="utf-8") as stream:
                stream.write(_formatter.format(record) + "\n")
        # handle() bypasses isEnabledFor()/logging.disable(); the dedicated
        # name also avoids inheriting the original modules' effective levels.
        logger.disabled = False
        logger.handle(record)
