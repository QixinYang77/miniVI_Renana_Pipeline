from dataclasses import dataclass
from typing import Optional, Tuple


def connect_ssh(*, host: str, username: str, key_filename: str):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, key_filename=key_filename)
    return client


def run_ssh(ssh_client, command: str, *, timeout_s: Optional[int] = None) -> Tuple[int, str, str]:
    stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout_s)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_status, out, err
