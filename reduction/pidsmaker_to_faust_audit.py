#!/usr/bin/env python3
"""Convert PIDSMaker CSV exports into FAUST-friendly auditd-style logs.

This script targets the raw CSV export layout currently present at:

    reduction/datasets/e3_raw/

Expected files:
    - edges.csv
    - subject_nodes.csv
    - file_nodes.csv
    - netflow_nodes.csv

The output is not intended to be a byte-for-byte Linux audit log replay.
Instead, it is a synthetic auditd-style log whose structure matches the subset
of records that FAUST's parser understands in this repository:

    - type=SYSCALL
    - type=CWD
    - type=PATH
    - type=SOCKADDR
    - type=PROCTITLE
    - type=EOE

Important PIDSMaker -> FAUST mapping choices
============================================

1. EVENT_OPEN -> open(2)-shaped audit event
   PIDSMaker edge direction is file -> subject. FAUST's open handler also
   models this as inode -> process, so the mapping is direct.

2. EVENT_READ -> read(2)-shaped audit event
   PIDSMaker edge direction is file -> subject. FAUST's read handler expects a
   file descriptor, so the converter reuses or synthesizes an earlier open().

3. EVENT_WRITE -> write(2)-shaped audit event
   PIDSMaker edge direction is subject -> file. FAUST again expects a file
   descriptor, so the converter reuses or synthesizes an earlier open().

4. EVENT_UNLINK / EVENT_UNLINKAT -> unlink(2)-shaped audit event
   PIDSMaker edge direction is subject -> file. FAUST's unlink handler creates
   an inode -> process relation, so we emit PATH metadata for the file being
   deleted.

5. EVENT_CONNECT / EVENT_ACCEPT -> connect/accept-shaped audit event
   These set up a synthetic socket FD for the process so that later
   send/receive events are acceptable to FAUST.

6. EVENT_SENDTO / EVENT_SENDMSG -> sendto(2)-shaped audit event
   PIDSMaker direction is subject -> netflow. FAUST models this as socket ->
   process (generated-by), which matches a send-like syscall.

7. EVENT_RECVFROM / EVENT_RECVMSG -> recvfrom(2)-shaped audit event
   PIDSMaker direction is netflow -> subject. FAUST models this as process ->
   socket (used), which matches a recv-like syscall.

8. EVENT_CLONE -> execve(2)-shaped parent/child audit event
   This is intentionally non-literal. In this FAUST codebase, the handler for
   syscall 59 (execve) is the supported path that constructs process lineage
   edges using pid/ppid. PIDSMaker's EVENT_CLONE already carries subject ->
   subject parent/child structure, so we emit an execve-shaped record to feed
   FAUST's supported parent/child path.

9. EVENT_EXECUTE -> open(2)-shaped executable file access
   This is also intentional. In this FAUST codebase, syscall 59 is used for
   process-parent lineage, not executable-file usage. PIDSMaker's
   EVENT_EXECUTE is file -> subject, so we map it to an open-shaped event to
   preserve the file -> process dependency that FAUST can understand.

Unsupported operations are skipped and counted in the summary.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


# x86_64 syscall numbers expected by FAUST's parser
SYS_READ = 0
SYS_WRITE = 1
SYS_OPEN = 2
SYS_CLOSE = 3
SYS_ACCEPT = 43
SYS_CONNECT = 42
SYS_SENDTO = 44
SYS_RECVFROM = 45
SYS_UNLINK = 87
SYS_EXECVE = 59
SYS_UNLINKAT = 263

ARCH_X86_64 = "c000003e"


@dataclass(frozen=True)
class SubjectInfo:
    index_id: int
    path: str
    cmd: str


@dataclass(frozen=True)
class FileInfo:
    index_id: int
    path: str


@dataclass(frozen=True)
class NetflowInfo:
    index_id: int
    src_addr: str
    src_port: str
    dst_addr: str
    dst_port: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reduction/datasets/e3_raw"),
        help="Directory containing PIDSMaker CSV export files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the converted FAUST-friendly audit log.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional limit for debugging or quick sample generation.",
    )
    return parser.parse_args()


def default_output_path(input_dir: Path) -> Path:
    return input_dir.with_suffix(".faust.log")


def load_subjects(path: Path) -> Dict[int, SubjectInfo]:
    subjects: Dict[int, SubjectInfo] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index_id = int(row["index_id"])
            subjects.setdefault(
                index_id,
                SubjectInfo(
                    index_id=index_id,
                    path=normalize_path(row["path"], fallback=f"/subject/{index_id}"),
                    cmd=(row["cmd"] or row["path"] or f"subject-{index_id}"),
                ),
            )
    return subjects


def load_files(path: Path) -> Dict[int, FileInfo]:
    files: Dict[int, FileInfo] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index_id = int(row["index_id"])
            files.setdefault(
                index_id,
                FileInfo(
                    index_id=index_id,
                    path=normalize_path(row["path"], fallback=f"/file/{index_id}"),
                ),
            )
    return files


def load_netflows(path: Path) -> Dict[int, NetflowInfo]:
    netflows: Dict[int, NetflowInfo] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index_id = int(row["index_id"])
            netflows.setdefault(
                index_id,
                NetflowInfo(
                    index_id=index_id,
                    src_addr=(row["src_addr"] or "NA"),
                    src_port=(row["src_port"] or "0"),
                    dst_addr=(row["dst_addr"] or "NA"),
                    dst_port=(row["dst_port"] or "0"),
                ),
            )
    return netflows


def normalize_path(raw: str, fallback: str) -> str:
    path = (raw or "").strip()
    if not path or path == "N/A":
        return fallback
    return path


def sanitize_token(text: str) -> str:
    # FAUST tokenizes by spaces, so we remove spaces from fields that must stay
    # within one token in the synthetic audit log.
    return (text or "").replace(" ", "_").replace('"', "_")


def quote_value(text: str) -> str:
    return f'"{sanitize_token(text)}"'


def proctitle_hex(text: str) -> str:
    payload = text or "unknown"
    return payload.encode("utf-8", "replace").hex()


def cwd_for_path(path: str) -> str:
    directory = os.path.dirname(path)
    return directory if directory else "."


def audit_stamp(timestamp_ns: int, serial: int) -> str:
    seconds = timestamp_ns // 1_000_000_000
    micros = (timestamp_ns % 1_000_000_000) // 1_000
    return f"audit({seconds}.{micros:06d}:{serial})"


def hex_fd(fd: int) -> str:
    return format(fd, "x")


class Converter:
    def __init__(
        self,
        subjects: Dict[int, SubjectInfo],
        files: Dict[int, FileInfo],
        netflows: Dict[int, NetflowInfo],
    ) -> None:
        self.subjects = subjects
        self.files = files
        self.netflows = netflows

        self.serial = 0
        self.parent_by_pid: Dict[int, int] = {}
        self.next_fd_by_pid: Dict[int, int] = defaultdict(lambda: 3)
        self.file_fd_by_pid: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.socket_fd_by_pid: Dict[int, Dict[int, int]] = defaultdict(dict)

        self.converted = Counter()
        self.skipped = Counter()
        self.synthetic = Counter()

    def next_serial(self) -> int:
        self.serial += 1
        return self.serial

    def allocate_fd(self, pid: int) -> int:
        fd = self.next_fd_by_pid[pid]
        self.next_fd_by_pid[pid] += 1
        return fd

    def subject_for(self, pid: int) -> SubjectInfo:
        return self.subjects.get(
            pid,
            SubjectInfo(index_id=pid, path=f"/subject/{pid}", cmd=f"subject-{pid}"),
        )

    def ensure_file_fd(
        self,
        pid: int,
        file_info: FileInfo,
        timestamp_ns: int,
        output,
    ) -> int:
        existing = self.file_fd_by_pid[pid].get(file_info.index_id)
        if existing is not None:
            return existing

        fd = self.allocate_fd(pid)
        self.file_fd_by_pid[pid][file_info.index_id] = fd
        self.synthetic["open_for_rw"] += 1

        subject = self.subject_for(pid)
        self.write_open_sequence(
            output=output,
            timestamp_ns=timestamp_ns,
            pid=pid,
            subject=subject,
            file_info=file_info,
            fd=fd,
            exe_field=subject.path,
        )
        return fd

    def ensure_socket_fd(self, pid: int, netflow: NetflowInfo) -> int:
        existing = self.socket_fd_by_pid[pid].get(netflow.index_id)
        if existing is not None:
            return existing

        fd = self.allocate_fd(pid)
        self.socket_fd_by_pid[pid][netflow.index_id] = fd
        return fd

    def kind_for_index(self, index_id: int) -> str:
        if index_id in self.subjects:
            return "subject"
        if index_id in self.files:
            return "file"
        if index_id in self.netflows:
            return "netflow"
        return "unknown"

    def emit_sequence(
        self,
        output,
        timestamp_ns: int,
        lines: Iterable[str],
    ) -> None:
        stamp = audit_stamp(timestamp_ns, self.next_serial())
        for line in lines:
            output.write(f"{line.format(stamp=stamp)}\n")

    def base_syscall_fields(
        self,
        syscall_no: int,
        pid: int,
        ppid: Optional[int] = None,
        a0: Optional[str] = None,
        exit_value: Optional[str] = None,
        exe: Optional[str] = None,
    ) -> str:
        parts = [
            "type=SYSCALL",
            "msg={stamp}:",
            f"arch={ARCH_X86_64}",
            f"syscall={syscall_no}",
            "success=yes",
            f"pid={pid}",
            "sid=0",
        ]
        if ppid is not None:
            parts.append(f"ppid={ppid}")
        if a0 is not None:
            parts.append(f"a0={a0}")
        if exit_value is not None:
            parts.append(f"exit={exit_value}")
        if exe is not None:
            parts.append(f"exe={quote_value(exe)}")
        return " ".join(parts)

    def write_open_sequence(
        self,
        output,
        timestamp_ns: int,
        pid: int,
        subject: SubjectInfo,
        file_info: FileInfo,
        fd: int,
        exe_field: Optional[str],
    ) -> None:
        self.emit_sequence(
            output,
            timestamp_ns,
            [
                self.base_syscall_fields(
                    syscall_no=SYS_OPEN,
                    pid=pid,
                    ppid=self.parent_by_pid.get(pid, 0),
                    exit_value=hex_fd(fd),
                    exe=exe_field,
                ),
                f"type=CWD msg={{stamp}}: cwd={quote_value(cwd_for_path(file_info.path))}",
                f"type=PATH msg={{stamp}}: name={quote_value(file_info.path)} inode={file_info.index_id}",
                f"type=PROCTITLE msg={{stamp}}: proctitle={proctitle_hex(subject.cmd)}",
                "type=EOE msg={stamp}:",
            ],
        )

    def write_file_io_sequence(
        self,
        output,
        timestamp_ns: int,
        syscall_no: int,
        pid: int,
        subject: SubjectInfo,
        fd: int,
        object_path: str,
    ) -> None:
        self.emit_sequence(
            output,
            timestamp_ns,
            [
                self.base_syscall_fields(
                    syscall_no=syscall_no,
                    pid=pid,
                    ppid=self.parent_by_pid.get(pid, 0),
                    a0=hex_fd(fd),
                    exit_value="1",
                    exe=object_path,
                ),
                f"type=PROCTITLE msg={{stamp}}: proctitle={proctitle_hex(subject.cmd)}",
                "type=EOE msg={stamp}:",
            ],
        )

    def write_unlink_sequence(
        self,
        output,
        timestamp_ns: int,
        syscall_no: int,
        pid: int,
        subject: SubjectInfo,
        file_info: FileInfo,
    ) -> None:
        self.emit_sequence(
            output,
            timestamp_ns,
            [
                self.base_syscall_fields(
                    syscall_no=syscall_no,
                    pid=pid,
                    ppid=self.parent_by_pid.get(pid, 0),
                    exe=subject.path,
                ),
                f"type=CWD msg={{stamp}}: cwd={quote_value(cwd_for_path(file_info.path))}",
                f"type=PATH msg={{stamp}}: name={quote_value(file_info.path)} inode={file_info.index_id}",
                f"type=PROCTITLE msg={{stamp}}: proctitle={proctitle_hex(subject.cmd)}",
                "type=EOE msg={stamp}:",
            ],
        )

    def write_execve_like_clone_sequence(
        self,
        output,
        timestamp_ns: int,
        parent_pid: int,
        child_pid: int,
        child: SubjectInfo,
    ) -> None:
        self.emit_sequence(
            output,
            timestamp_ns,
            [
                self.base_syscall_fields(
                    syscall_no=SYS_EXECVE,
                    pid=child_pid,
                    ppid=parent_pid,
                    exit_value="0",
                    exe=child.path,
                ),
                f"type=PROCTITLE msg={{stamp}}: proctitle={proctitle_hex(child.cmd)}",
                "type=EOE msg={stamp}:",
            ],
        )

    def write_socket_sequence(
        self,
        output,
        timestamp_ns: int,
        syscall_no: int,
        pid: int,
        subject: SubjectInfo,
        fd: int,
        netflow: NetflowInfo,
    ) -> None:
        self.emit_sequence(
            output,
            timestamp_ns,
            [
                self.base_syscall_fields(
                    syscall_no=syscall_no,
                    pid=pid,
                    ppid=self.parent_by_pid.get(pid, 0),
                    a0=hex_fd(fd),
                    exit_value="1",
                    exe=subject.path,
                ),
                f"type=SOCKADDR msg={{stamp}}: saddr={self.encode_sockaddr(netflow)}",
                f"type=PROCTITLE msg={{stamp}}: proctitle={proctitle_hex(subject.cmd)}",
                "type=EOE msg={stamp}:",
            ],
        )

    @staticmethod
    def encode_sockaddr(netflow: NetflowInfo) -> str:
        return sanitize_token(
            f"{netflow.src_addr}:{netflow.src_port}->{netflow.dst_addr}:{netflow.dst_port}"
        )

    def convert_edge(self, row: Dict[str, str], output) -> None:
        op = row["operation"]
        timestamp_ns = int(row["timestamp_rec"])
        src_index = int(row["src_index_id"])
        dst_index = int(row["dst_index_id"])
        src_kind = self.kind_for_index(src_index)
        dst_kind = self.kind_for_index(dst_index)

        if op == "EVENT_CLONE" and src_kind == "subject" and dst_kind == "subject":
            parent_pid = src_index
            child_pid = dst_index
            self.parent_by_pid[child_pid] = parent_pid
            self.write_execve_like_clone_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                parent_pid=parent_pid,
                child_pid=child_pid,
                child=self.subject_for(child_pid),
            )
            self.converted[op] += 1
            return

        if op in {"EVENT_OPEN", "EVENT_EXECUTE"} and src_kind == "file" and dst_kind == "subject":
            file_info = self.files[src_index]
            pid = dst_index
            subject = self.subject_for(pid)
            fd = self.allocate_fd(pid)
            self.file_fd_by_pid[pid][file_info.index_id] = fd

            # EVENT_EXECUTE is emitted as an open-like event on purpose. FAUST's
            # supported execve path models process lineage, not executable-file
            # access, while EVENT_EXECUTE is file -> subject.
            exe_field = subject.path if op == "EVENT_OPEN" else file_info.path
            self.write_open_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                pid=pid,
                subject=subject,
                file_info=file_info,
                fd=fd,
                exe_field=exe_field,
            )
            self.converted[op] += 1
            return

        if op == "EVENT_READ" and src_kind == "file" and dst_kind == "subject":
            file_info = self.files[src_index]
            pid = dst_index
            subject = self.subject_for(pid)
            fd = self.ensure_file_fd(pid, file_info, timestamp_ns, output)
            self.write_file_io_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                syscall_no=SYS_READ,
                pid=pid,
                subject=subject,
                fd=fd,
                object_path=file_info.path,
            )
            self.converted[op] += 1
            return

        if op == "EVENT_WRITE" and src_kind == "subject" and dst_kind == "file":
            pid = src_index
            file_info = self.files[dst_index]
            subject = self.subject_for(pid)
            fd = self.ensure_file_fd(pid, file_info, timestamp_ns, output)
            self.write_file_io_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                syscall_no=SYS_WRITE,
                pid=pid,
                subject=subject,
                fd=fd,
                object_path=file_info.path,
            )
            self.converted[op] += 1
            return

        if op in {"EVENT_UNLINK", "EVENT_UNLINKAT"} and src_kind == "subject" and dst_kind == "file":
            pid = src_index
            subject = self.subject_for(pid)
            file_info = self.files[dst_index]
            self.write_unlink_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                syscall_no=SYS_UNLINKAT if op == "EVENT_UNLINKAT" else SYS_UNLINK,
                pid=pid,
                subject=subject,
                file_info=file_info,
            )
            self.converted[op] += 1
            return

        if op in {"EVENT_CONNECT", "EVENT_ACCEPT"} and {
            src_kind,
            dst_kind,
        } == {"subject", "netflow"}:
            pid, netflow_index = (
                (src_index, dst_index) if src_kind == "subject" else (dst_index, src_index)
            )
            subject = self.subject_for(pid)
            netflow = self.netflows[netflow_index]
            fd = self.ensure_socket_fd(pid, netflow)
            self.write_socket_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                syscall_no=SYS_ACCEPT if op == "EVENT_ACCEPT" else SYS_CONNECT,
                pid=pid,
                subject=subject,
                fd=fd,
                netflow=netflow,
            )
            self.converted[op] += 1
            return

        if op in {"EVENT_SENDTO", "EVENT_SENDMSG"} and {
            src_kind,
            dst_kind,
        } == {"subject", "netflow"}:
            pid, netflow_index = (
                (src_index, dst_index) if src_kind == "subject" else (dst_index, src_index)
            )
            subject = self.subject_for(pid)
            netflow = self.netflows[netflow_index]
            fd = self.ensure_socket_fd(pid, netflow)
            self.write_socket_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                syscall_no=SYS_SENDTO,
                pid=pid,
                subject=subject,
                fd=fd,
                netflow=netflow,
            )
            self.converted[op] += 1
            return

        if op in {"EVENT_RECVFROM", "EVENT_RECVMSG"} and {
            src_kind,
            dst_kind,
        } == {"subject", "netflow"}:
            pid, netflow_index = (
                (src_index, dst_index) if src_kind == "subject" else (dst_index, src_index)
            )
            subject = self.subject_for(pid)
            netflow = self.netflows[netflow_index]
            fd = self.ensure_socket_fd(pid, netflow)
            self.write_socket_sequence(
                output=output,
                timestamp_ns=timestamp_ns,
                syscall_no=SYS_RECVFROM,
                pid=pid,
                subject=subject,
                fd=fd,
                netflow=netflow,
            )
            self.converted[op] += 1
            return

        if op == "EVENT_CLOSE" and {src_kind, dst_kind} == {"subject", "file"}:
            pid, file_index = (
                (src_index, dst_index) if src_kind == "subject" else (dst_index, src_index)
            )
            subject = self.subject_for(pid)
            file_info = self.files[file_index]
            fd = self.file_fd_by_pid[pid].get(file_index)
            if fd is None:
                self.skipped[f"{op}:missing_fd"] += 1
                return
            self.emit_sequence(
                output,
                timestamp_ns,
                [
                    self.base_syscall_fields(
                        syscall_no=SYS_CLOSE,
                        pid=pid,
                        ppid=self.parent_by_pid.get(pid, 0),
                        a0=hex_fd(fd),
                        exit_value="0",
                        exe=file_info.path,
                    ),
                    f"type=PROCTITLE msg={{stamp}}: proctitle={proctitle_hex(subject.cmd)}",
                    "type=EOE msg={stamp}:",
                ],
            )
            del self.file_fd_by_pid[pid][file_index]
            self.converted[op] += 1
            return

        self.skipped[op] += 1


def run_conversion(input_dir: Path, output_path: Path, max_edges: Optional[int]) -> Converter:
    edges_path = input_dir / "edges.csv"
    subjects = load_subjects(input_dir / "subject_nodes.csv")
    files = load_files(input_dir / "file_nodes.csv")
    netflows = load_netflows(input_dir / "netflow_nodes.csv")

    converter = Converter(subjects=subjects, files=files, netflows=netflows)

    with edges_path.open(newline="") as edge_handle, output_path.open("w") as output:
        reader = csv.DictReader(edge_handle)
        for edge_count, row in enumerate(reader, 1):
            converter.convert_edge(row, output)
            if max_edges is not None and edge_count >= max_edges:
                break

    return converter


def print_summary(converter: Converter, output_path: Path) -> None:
    print(f"Wrote converted audit log to: {output_path}")
    print("Converted operations:")
    for op, count in sorted(converter.converted.items()):
        print(f"  {op}: {count}")

    if converter.synthetic:
        print("Synthetic helper events inserted:")
        for name, count in sorted(converter.synthetic.items()):
            print(f"  {name}: {count}")

    if converter.skipped:
        print("Skipped operations:")
        for op, count in sorted(converter.skipped.items()):
            print(f"  {op}: {count}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output or default_output_path(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = run_conversion(
        input_dir=input_dir,
        output_path=output_path,
        max_edges=args.max_edges,
    )
    print_summary(converter, output_path)


if __name__ == "__main__":
    main()
