import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_IMAGES = [
    "rkader2811/smart-office",
    "rkader2811/warehouse",
    "rkader2811/music-platform",
]
DEDICATED_PORTS = [3001, 5173, 8000, 8082]
COMPOSE_BASE_COMMAND = ["docker", "compose", "--profile", "platform"]
COMPOSE_STOP_SERVICES = ["platform", "session-db", "backend", "frontend"]


def parse_args():
    parser = argparse.ArgumentParser(description="Manage the local SAGE dev runtime.")
    parser.add_argument("action", choices=["stop", "soft-cleanup", "hard-cleanup"])
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing state.")
    return parser.parse_args()


def run_command(command, dry_run=False, check=True):
    print(f"+ {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=SRC_ROOT, check=check)


def list_benchmark_containers():
    output = subprocess.check_output(
        ["docker", "ps", "-a", "--format", "{{.ID}} {{.Image}} {{.Status}}"],
        cwd=SRC_ROOT,
        text=True,
    )
    containers = []
    for line in output.splitlines():
        container_id, image_name, status = line.split(" ", 2)
        if image_name in BENCHMARK_IMAGES:
            containers.append((container_id, image_name, status))
    return containers


def remove_benchmark_containers(dry_run=False):
    for container_id, image_name, status in list_benchmark_containers():
        print(f"+ docker rm -f {container_id}  # {image_name} [{status}]")
        if not dry_run:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                cwd=SRC_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def listening_socket_inodes():
    inode_to_port = {}
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(proc_file).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            port = int(parts[1].split(":")[1], 16)
            if port in DEDICATED_PORTS:
                inode_to_port[parts[9]] = port
    return inode_to_port


def port_listener_pids():
    inode_to_port = listening_socket_inodes()
    pid_to_port = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                target = os.readlink(fd)
                if not target.startswith("socket:["):
                    continue
                inode = target[8:-1]
                if inode in inode_to_port:
                    pid_to_port[int(proc.name)] = inode_to_port[inode]
                    break
        except OSError:
            continue
    return pid_to_port


def describe_pid(pid):
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode().strip()
        return cmdline or Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def kill_port_listeners(dry_run=False):
    pid_to_port = port_listener_pids()
    if not pid_to_port:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        remaining = {}
        for pid, port in sorted(pid_to_port.items()):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            print(f"+ kill -{sig.value} {pid}  # port {port} {describe_pid(pid)}")
            if not dry_run:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    continue
            remaining[pid] = port
        if dry_run or sig == signal.SIGKILL:
            return
        time.sleep(2)
        pid_to_port = {pid: port for pid, port in remaining.items() if Path(f"/proc/{pid}").exists()}
        if not pid_to_port:
            return


def stop_runtime(dry_run=False):
    remove_benchmark_containers(dry_run=dry_run)
    run_command(COMPOSE_BASE_COMMAND + ["stop", *COMPOSE_STOP_SERVICES], dry_run=dry_run, check=False)
    kill_port_listeners(dry_run=dry_run)


def main():
    args = parse_args()
    stop_runtime(dry_run=args.dry_run)
    if args.action == "soft-cleanup":
        run_command(COMPOSE_BASE_COMMAND + ["down", "--remove-orphans"], dry_run=args.dry_run, check=False)
    elif args.action == "hard-cleanup":
        run_command(COMPOSE_BASE_COMMAND + ["down", "--volumes", "--remove-orphans"], dry_run=args.dry_run, check=False)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)

