import subprocess
import socket
import time
from openai import OpenAI
from base import InferenceEngine
from registry import register
import threading

def wait_for_port(host: str, port: int, timeout: int = 300):
    """Poll until the target port is ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"vLLM API server did not start on {host}:{port} within {timeout}s")

@register("vllm_api")
class VllmAPIProvider(InferenceEngine):
    def __init__(self, **kwargs):
        """
        init_kwargs:
            model_name: model name or local path
            host: API server host (default 127.0.0.1)
            port: API server port (default 8000)
            api_key: any string; vLLM API server does not validate it by default (default "EMPTY")
            request_timeout: per-request timeout (used by the client at call time)
        """
        self.model = kwargs["model_name"]
        print(f"Using model: {self.model}")
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 8000)
        self.api_key = kwargs.get("api_key", "EMPTY")
        self.request_timeout = kwargs.get("request_timeout", 300)

        self._start_server()
        wait_for_port(self.host, self.port, timeout=300)
        self._setup()

    def _start_server(self):
        """Start the vLLM OpenAI-compatible API server in background."""
        gpu_util = 0.8
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",  
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(gpu_util),

        ]

        log_file = open("vllm_api.log", "w")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        def log_thread(pipe, file):
            for line in pipe:
                print(line, end="")  # Print to console.
                file.write(line)
                file.flush()
        threading.Thread(target=log_thread, args=(self.proc.stdout, log_file), daemon=True).start()

    def _setup(self):
        """Initialize the OpenAI SDK client."""
        base_url = f"http://{self.host}:{self.port}/v1"
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

    def infer(self, prompts: list[str], **gen_kwargs) -> list[str]:
        """Run inference through the /v1/completions endpoint."""
        responses = []
        for prompt in prompts:
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                **gen_kwargs
            )
            responses.append(resp.choices[0].text)
            print(resp.choices[0].text)
        return responses

    def __del__(self):
        """Clean up the background process on exit."""
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
