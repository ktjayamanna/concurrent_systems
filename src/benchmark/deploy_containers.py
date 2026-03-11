import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request


SAFE_BENCHMARK_API_PORT = 28082
BENCHMARK_CONTAINERS = [
    "smart-office",
    "warehouse",
    "music-platform",
]
LEGACY_BENCHMARK_CONTAINERS = [
    "rkader2811/smart-office",
    "rkader2811/warehouse",
    "rkader2811/music-platform",
]
BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "opaca-llm-benchmark-containers" / "scripts" / "build_containers.py"
BUILD_SCRIPT_CWD = BUILD_SCRIPT.parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform-url", required=True, help="Base URL of the OPACA platform.")
    return parser.parse_args()


def http_json(method, url, payload=None, timeout_seconds=10):
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            if not body:
                return None
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {body or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def wait_for_platform(platform_url, timeout_seconds=60):
    deadline = time.time() + timeout_seconds
    info_url = f"{platform_url.rstrip('/')}/info"
    while time.time() < deadline:
        try:
            http_json("GET", info_url, timeout_seconds=5)
            print(f"Platform ready at {platform_url}")
            return
        except RuntimeError:
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for OPACA platform at {platform_url}")


def build_local_benchmark_images():
    subprocess.run(["python3", str(BUILD_SCRIPT)], cwd=BUILD_SCRIPT_CWD, check=True)


def remove_stale_docker_containers():
    managed_images = set(BENCHMARK_CONTAINERS) | set(LEGACY_BENCHMARK_CONTAINERS)
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.ID}} {{.Image}} {{.Status}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        container_id, image_name, status = line.split(" ", 2)
        if image_name not in managed_images:
            continue
        if status.startswith("Up "):
            continue
        subprocess.run(["docker", "rm", "-f", container_id], check=True, capture_output=True, text=True)
        print(f"Removed stale Docker container: {image_name} ({container_id})")


def sync_containers(platform_url):
    containers_url = f"{platform_url.rstrip('/')}/containers"
    managed_images = set(BENCHMARK_CONTAINERS) | set(LEGACY_BENCHMARK_CONTAINERS)

    existing = http_json("GET", containers_url, timeout_seconds=15) or []
    managed_existing = [
        container for container in existing
        if container.get("image", {}).get("imageName") in managed_images
    ]

    for container in managed_existing:
        container_id = container["containerId"]
        image_name = container["image"]["imageName"]
        http_json("DELETE", f"{containers_url}/{container_id}", {}, timeout_seconds=120)
        print(f"Removed managed container: {image_name} ({container_id})")

    for image_name in BENCHMARK_CONTAINERS:
        http_json(
            "POST",
            containers_url,
            {"image": {"imageName": image_name, "apiPort": SAFE_BENCHMARK_API_PORT}},
            timeout_seconds=600,
        )
        print(f"Deployed benchmark container: {image_name}")


def enhance_error_message(exc):
    message = str(exc)
    port_conflict = f"failed to bind host port 0.0.0.0:{SAFE_BENCHMARK_API_PORT}" in message
    if not port_conflict:
        return RuntimeError(message)
    hint = (
        f"Host port {SAFE_BENCHMARK_API_PORT} is already in use while OPACA tries to start a benchmark container. "
        f"In this devcontainer setup, the usual cause is VS Code auto-forwarding port {SAFE_BENCHMARK_API_PORT}. "
        f"Close any forwarded {SAFE_BENCHMARK_API_PORT} port (or reload the VS Code window after applying the devcontainer config) "
        "and retry."
    )
    return RuntimeError(f"{message}\nHint: {hint}")


def main():
    args = parse_args()
    build_local_benchmark_images()
    wait_for_platform(args.platform_url)
    remove_stale_docker_containers()
    try:
        sync_containers(args.platform_url)
    except Exception as exc:
        remove_stale_docker_containers()
        raise enhance_error_message(exc) from exc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Benchmark container deployment failed: {exc}", file=sys.stderr)
        sys.exit(1)