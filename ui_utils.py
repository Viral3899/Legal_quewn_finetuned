#!/usr/bin/env python3
"""Lightweight CLI UI helpers: toasts and loading spinners.

Usage:
    from ui_utils import toast, loading

    toast("Model loaded", type="success")
    with loading("Loading model..."):
        do_work()
"""

import sys
import threading
import time
from contextlib import contextmanager


TOAST_PREFIX = {
    "info": "ℹ️ ",
    "success": "✅ ",
    "warning": "⚠️ ",
    "error": "❌ ",
}


def toast(message: str, type: str = "info") -> None:
    prefix = TOAST_PREFIX.get(type, TOAST_PREFIX["info"])
    print(f"{prefix}{message}")


class _Spinner:
    def __init__(self, text: str = "Loading...") -> None:
        self.text = text
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def run():
            frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            idx = 0
            while not self._stop_event.is_set():
                frame = frames[idx % len(frames)]
                sys.stdout.write(f"\r{frame} {self.text}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.09)
            # clear line
            sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
            sys.stdout.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None


@contextmanager
def loading(text: str = "Loading..."):
    spinner = _Spinner(text)
    try:
        spinner.start()
        yield
    finally:
        spinner.stop()


