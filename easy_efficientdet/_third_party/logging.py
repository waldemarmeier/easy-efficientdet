# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
import logging
import sys
from threading import Lock

_logger = None
_logger_lock = Lock()
DEFAULT_LOG_LEVEL_INTERACTIVE = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
BASE_LOGGER_NAME = "easy-efficientdet"


# tensorflow logging setup is quiet nice so I use it here
def get_logger(name: str):
    """Return TF logger instance."""
    global _logger

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        # Scope the TensorFlow logger to not conflict with users' loggers.
        logger = logging.getLogger(BASE_LOGGER_NAME)

        # Override findCaller on the logger to skip internal helper functions
        #     logger.findCaller = _logger_find_caller

        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if sys.ps1:
                    _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = sys.flags.interactive

            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel(DEFAULT_LOG_LEVEL_INTERACTIVE)
                _logging_target = sys.stdout
            else:
                _logging_target = sys.stderr

            # Add the output handler.
            _handler = logging.StreamHandler(_logging_target)
            _handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, None))
            logger.addHandler(_handler)

        _logger = logger
        return _logger.getChild(name)

    finally:
        _logger_lock.release()
