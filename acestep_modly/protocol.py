"""Stream native/Python logs as Modly JSON lines, with one terminal outcome."""
import contextlib
import json
import os
import re
import sys
import threading


def configure_stdio():
    """Keep Unicode native/upstream logs safe under legacy Windows codepages."""
    os.environ.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8:replace", "PYTHONUNBUFFERED": "1"})
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


class Protocol:
    def __init__(self):
        self.output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()
        self.percent = 0
        self.terminal = False

    def emit(self, message):
        with self.lock:
            if self.terminal:
                return
            if message["type"] in ("done", "error"):
                self.terminal = True
            self.output.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.output.flush()

    def log(self, message):
        # Redact token-looking strings in exception/log output, never read tokens from settings.
        message = re.sub(r"hf_[A-Za-z0-9]{15,}", "[REDACTED]", str(message))
        self.emit({"type": "log", "message": message[:12000]})

    def progress(self, percent, label):
        self.percent = max(self.percent, min(100, int(percent)))
        self.emit({"type": "progress", "percent": self.percent, "label": str(label)})

    @contextlib.contextmanager
    def capture_logs(self):
        """Capture fd 1/2 as well as Python prints; stderr is not live in upstream Modly."""
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        reader, writer = os.pipe()

        def forward():
            with os.fdopen(reader, "r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if line.strip():
                        self.log(line.rstrip())

        thread = threading.Thread(target=forward, daemon=True)
        thread.start()
        os.dup2(writer, 1)
        os.dup2(writer, 2)
        os.close(writer)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            thread.join(timeout=5)

    @contextlib.contextmanager
    def heartbeat(self):
        stop = threading.Event()

        def report():
            while not stop.wait(20):
                self.log("ACE-Step is still running; model loading/LM planning can take several minutes.")

        thread = threading.Thread(target=report, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1)
