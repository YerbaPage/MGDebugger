import json
import os
from threading import Thread
import resource
import signal
import json
import os
import traceback
from loguru import logger


def set_memory_limit(max_memory):
    def limit_memory():
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    return limit_memory


def timeout_handler(_, __):
    raise TimeoutError()


def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        if self.is_alive():
            return None
        return self.ret

    def terminate(self):
        self._stop()


def function_with_timeout(func, args, timeout, max_memory=100 * 1024 * 1024):
    result_container = []

    def wrapper():
        # set_memory_limit(max_memory)()
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.error(f"Timeout Error\n {args[0]} with timeout {timeout}")
        thread.terminate()
        raise TimeoutError()
    else:
        return result_container[0]
