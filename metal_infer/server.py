#!/usr/bin/env python3
"""
OpenAI-compatible API server for Flash-MoE inference engine.

Exposes /v1/chat/completions with streaming SSE support.
Wraps the C/Metal inference engine via subprocess.

Usage:
    cd metal_infer
    uv run server.py [--port 8000] [--2bit]

Then point any OpenAI-compatible client at http://localhost:8000
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

# Path to inference binary
INFER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer")
ENCODE_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encode_prompt.py")

MODEL_NAME = "qwen3.5-397b-a17b"
USE_2BIT = False


def format_messages(messages, tools=None):
    """Format OpenAI messages array into Qwen3 chat template string."""
    parts = []

    # If tools are provided, add them to the system prompt
    tool_text = ""
    if tools:
        tool_defs = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                })
        if tool_defs:
            tool_text = (
                "\n\n# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>\n"
                + "\n".join(json.dumps(t) for t in tool_defs)
                + "\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>"
            )

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"<|im_start|>system\n{content}{tool_text}<|im_end|>")
            tool_text = ""  # only add tools once
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            if msg.get("tool_calls"):
                # Format tool calls
                tc_text = ""
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    tc_text += (
                        f"<tool_call>\n"
                        f'{{"name": "{fn["name"]}", "arguments": {fn["arguments"]}}}\n'
                        f"</tool_call>"
                    )
                parts.append(f"<|im_start|>assistant\n{tc_text}<|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            name = msg.get("name", "")
            parts.append(
                f"<|im_start|>user\n"
                f"<tool_response>\n{content}\n</tool_response><|im_end|>"
            )

    # Add system prompt with /think if none provided
    if not any(m["role"] == "system" for m in messages):
        sys_content = "You are a helpful assistant. /think"
        parts.insert(0, f"<|im_start|>system\n{sys_content}{tool_text}<|im_end|>")

    # Add assistant prompt
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def run_inference(prompt_text, max_tokens=1024, stream_callback=None):
    """Run inference and yield tokens."""
    # Write prompt to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt_text)
        text_path = f.name

    tok_path = text_path + ".bin"

    try:
        # Tokenize
        cmd = f'python3 {ENCODE_BIN} "$(cat {text_path})" -o {tok_path}'
        rc = subprocess.run(cmd, shell=True, capture_output=True)
        if rc.returncode != 0:
            yield {"error": f"Tokenization failed: {rc.stderr.decode()[:200]}"}
            return

        # Run inference
        infer_cmd = [
            INFER_BIN,
            "--prompt-tokens", tok_path,
            "--tokens", str(max_tokens),
            "--cache-entries", "0",
        ]
        if USE_2BIT:
            infer_cmd.append("--2bit")

        proc = subprocess.Popen(
            infer_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        full_text = ""
        for line in proc.stderr:
            # Parse token output lines like: " hello  [gen 1/20] token_id=123 (100 ms, 10.0 tok/s)"
            if "[gen " in line and "token_id=" in line:
                # Extract the text before [gen
                text_part = line.split("[gen")[0].rstrip()
                if text_part:
                    # Handle special tokens
                    if "<think>" in text_part or "</think>" in text_part:
                        continue  # skip thinking tokens in API output
                    if "<|im_end|>" in text_part or "<|im_start|>" in text_part:
                        break
                    full_text += text_part
                    if stream_callback:
                        stream_callback(text_part)

        proc.wait()
        yield {"text": full_text}

    finally:
        os.unlink(text_path)
        if os.path.exists(tok_path):
            os.unlink(tok_path)


def parse_tool_calls(text):
    """Parse <tool_call>...</tool_call> blocks from model output."""
    tool_calls = []
    import re
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {})),
                }
            })
        except json.JSONDecodeError:
            pass
    return tool_calls


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        sys.stderr.write(f"[api] {args[0]}\n")

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "owned_by": "local",
                }]
            })
        elif self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        tools = body.get("tools", None)
        max_tokens = body.get("max_tokens", 1024)
        stream = body.get("stream", False)

        prompt_text = format_messages(messages, tools)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            full_text = ""

            def on_token(text):
                nonlocal full_text
                full_text += text
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            for result in run_inference(prompt_text, max_tokens, on_token):
                if "error" in result:
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"[Error: {result['error']}]"},
                            "finish_reason": "stop",
                        }]
                    }
                    self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())

            # Check for tool calls in the full text
            tool_calls = parse_tool_calls(full_text) if tools else []

            # Send final chunk
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }]
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            # Non-streaming
            full_text = ""
            for result in run_inference(prompt_text, max_tokens):
                if "text" in result:
                    full_text = result["text"]
                elif "error" in result:
                    self.send_json({"error": result["error"]}, status=500)
                    return

            # Check for tool calls
            tool_calls = parse_tool_calls(full_text) if tools else []

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text if not tool_calls else None,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }],
                "usage": {
                    "prompt_tokens": len(prompt_text.split()),  # approximate
                    "completion_tokens": len(full_text.split()),
                    "total_tokens": len(prompt_text.split()) + len(full_text.split()),
                }
            }
            if tool_calls:
                response["choices"][0]["message"]["tool_calls"] = tool_calls

            self.send_json(response)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()


def main():
    global USE_2BIT

    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for Flash-MoE")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--2bit", dest="use_2bit", action="store_true")
    args = parser.parse_args()

    USE_2BIT = args.use_2bit

    if not os.path.exists(INFER_BIN):
        print(f"Error: {INFER_BIN} not found. Run 'make infer' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Flash-MoE API Server")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Quant:    {'2-bit' if USE_2BIT else '4-bit'}")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Health:   http://{args.host}:{args.port}/health")
    print()
    print(f"Compatible with OpenAI SDK, curl, pi.dev, etc.")
    print(f"  export OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    print(f"  export OPENAI_API_KEY=local")
    print()

    server = HTTPServer((args.host, args.port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
