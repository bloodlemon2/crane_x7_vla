# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Slurmコマンドラッパーモジュール.

sbatch, squeue, scancelなどのSlurmコマンドをラップする。
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live

from lifter.clients.ssh import SSHClient
from lifter.config import SlurmConfig
from lifter.core.console import console
from lifter.ui.monitor import MonitorDisplayBuilder, MonitorState
from lifter.ui.status_table import print_job_status_table
from lifter.utils import format_duration_timer, parse_job_id

if TYPE_CHECKING:
    pass


class SlurmError(Exception):
    """Slurm操作に関するエラー."""


@dataclass
class JobInfo:
    """Slurmジョブ情報."""

    job_id: str
    name: str
    user: str
    state: str
    partition: str
    time: str
    nodes: int
    nodelist: str = ""
    reason: str = ""
    stdout_file: str = ""
    stderr_file: str = ""

    @property
    def nodelist_or_reason(self) -> str:
        """NODELIST(REASON)形式で返す."""
        if self.is_pending:
            return f"({self.reason})" if self.reason else ""
        return self.nodelist

    @property
    def is_running(self) -> bool:
        """ジョブが実行中かどうか."""
        return self.state in ("RUNNING", "R")

    @property
    def is_pending(self) -> bool:
        """ジョブが保留中かどうか."""
        return self.state in ("PENDING", "PD")

    @property
    def is_completed(self) -> bool:
        """ジョブが完了したかどうか."""
        return self.state in ("COMPLETED", "CD")

    @property
    def is_failed(self) -> bool:
        """ジョブが失敗したかどうか."""
        return self.state in ("FAILED", "F", "TIMEOUT", "TO", "CANCELLED", "CA", "NODE_FAIL", "NF")

    @property
    def is_active(self) -> bool:
        """ジョブがアクティブ（実行中または保留中）かどうか."""
        return self.is_running or self.is_pending


@dataclass
class LogLine:
    """ログ行."""

    text: str
    is_stderr: bool = False


@dataclass
class JobMonitor:
    """ジョブ監視状態を管理."""

    job_id: str
    start_time: float = field(default_factory=time.time)
    log_lines: deque = field(default_factory=lambda: deque(maxlen=100))
    last_stdout_offset: int = 0
    last_stderr_offset: int = 0
    state: str = "UNKNOWN"
    slurm_time: str = "0:00"

    @property
    def elapsed_time(self) -> str:
        """経過時間を取得."""
        return format_duration_timer(time.time() - self.start_time)

    def add_log_lines(self, lines: list[str], is_stderr: bool = False) -> None:
        """ログ行を追加."""
        for line in lines:
            stripped = line.rstrip()
            if stripped:
                self.log_lines.append(LogLine(text=stripped, is_stderr=is_stderr))


class SlurmClient:
    """Slurmコマンドのラッパー."""

    def __init__(self, ssh: SSHClient, config: SlurmConfig):
        """Slurmクライアントを初期化.

        Args:
            ssh: SSH接続クライアント
            config: Slurm設定
        """
        self.ssh = ssh
        self.config = config

    def submit(self, script_path: Path, remote_script_path: str | None = None) -> str:
        """ジョブスクリプトを投下.

        Args:
            script_path: ローカルのジョブスクリプトパス
            remote_script_path: リモートでのスクリプトパス (省略時は自動生成)

        Returns:
            投下されたジョブのID

        Raises:
            SlurmError: ジョブ投下に失敗した場合
        """
        if not script_path.exists():
            raise SlurmError(f"スクリプトファイルが見つかりません: {script_path}")

        # リモートパスを決定
        if remote_script_path is None:
            remote_workdir = str(self.config.remote_workdir).replace("$HOME", "~")
            remote_script_path = f"{remote_workdir}/jobs/{script_path.name}"

        # リモートディレクトリを作成
        remote_dir = str(Path(remote_script_path).parent)
        self.ssh.makedirs(remote_dir)

        # スクリプトをアップロード
        console.print(f"[dim]スクリプトをアップロード中: {remote_script_path}[/dim]")
        self.ssh.upload(script_path, remote_script_path)

        # sbatchで投下（カレントディレクトリから実行する必要がある）
        script_filename = Path(remote_script_path).name
        sbatch_cmd = f"cd {remote_dir} && sbatch {script_filename}"
        console.print(f"[dim]ジョブを投下中: {sbatch_cmd}[/dim]")
        stdout, stderr, exit_code = self.ssh.execute(sbatch_cmd)

        # sbatchの出力を表示
        if stdout.strip():
            console.print(f"[dim]{stdout.strip()}[/dim]")
        if stderr.strip():
            console.print(f"[yellow]{stderr.strip()}[/yellow]")

        if exit_code != 0:
            raise SlurmError(f"ジョブ投下に失敗しました: {stderr}")

        # ジョブIDを抽出
        job_id = parse_job_id(stdout)
        if job_id is None:
            raise SlurmError(f"ジョブIDを抽出できませんでした: {stdout}")

        console.print(f"[green]ジョブが投下されました: {job_id}[/green]")
        return job_id

    def submit_script_content(self, script_content: str, script_name: str = "job.sh") -> str:
        """ジョブスクリプト内容を直接投下.

        Args:
            script_content: ジョブスクリプトの内容
            script_name: リモートでのスクリプト名

        Returns:
            投下されたジョブのID

        Raises:
            SlurmError: ジョブ投下に失敗した場合
        """
        remote_workdir = str(self.config.remote_workdir).replace("$HOME", "~")
        remote_script_path = f"{remote_workdir}/jobs/{script_name}"

        # リモートディレクトリを作成
        remote_dir = str(Path(remote_script_path).parent)
        self.ssh.makedirs(remote_dir)

        # スクリプトをアップロード
        console.print(f"[dim]スクリプトをアップロード中: {remote_script_path}[/dim]")
        self.ssh.upload_string(script_content, remote_script_path)

        # 実行権限を付与
        self.ssh.execute(f"chmod +x {remote_script_path}")

        # sbatchで投下（カレントディレクトリから実行する必要がある）
        sbatch_cmd = f"cd {remote_dir} && sbatch {script_name}"
        console.print(f"[dim]ジョブを投下中: {sbatch_cmd}[/dim]")
        stdout, stderr, exit_code = self.ssh.execute(sbatch_cmd)

        # sbatchの出力を表示
        if stdout.strip():
            console.print(f"[dim]{stdout.strip()}[/dim]")
        if stderr.strip():
            console.print(f"[yellow]{stderr.strip()}[/yellow]")

        if exit_code != 0:
            raise SlurmError(f"ジョブ投下に失敗しました: {stderr}")

        # ジョブIDを抽出
        job_id = parse_job_id(stdout)
        if job_id is None:
            raise SlurmError(f"ジョブIDを抽出できませんでした: {stdout}")

        console.print(f"[green]ジョブが投下されました: {job_id}[/green]")
        return job_id

    def status(self, job_id: str | None = None, user: str | None = None) -> list[JobInfo]:
        """ジョブ状態を取得.

        Args:
            job_id: 特定のジョブID (省略時は全ジョブ)
            user: 特定のユーザー (省略時は自分のジョブ)

        Returns:
            ジョブ情報のリスト
        """
        # squeueコマンドを構築
        # %R: REASON (PENDINGの理由)
        cmd = "squeue --format='%i|%j|%u|%T|%P|%M|%D|%N|%R' --noheader"
        if job_id:
            cmd += f" --job={job_id}"
        if user:
            cmd += f" --user={user}"
        else:
            # デフォルトは自分のジョブ
            cmd += " --me"

        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code != 0:
            # ジョブが見つからない場合は空リストを返す
            if "Invalid job id" in stderr or "not found" in stderr.lower():
                return []
            raise SlurmError(f"squeue実行に失敗しました: {stderr}")

        jobs = []
        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 7:
                jobs.append(
                    JobInfo(
                        job_id=parts[0].strip(),
                        name=parts[1].strip(),
                        user=parts[2].strip(),
                        state=parts[3].strip(),
                        partition=parts[4].strip(),
                        time=parts[5].strip(),
                        nodes=int(parts[6].strip()) if parts[6].strip().isdigit() else 1,
                        nodelist=parts[7].strip() if len(parts) > 7 else "",
                        reason=parts[8].strip() if len(parts) > 8 else "",
                    )
                )

        return jobs

    def get_job_state(self, job_id: str) -> str | None:
        """特定ジョブの状態を取得.

        Args:
            job_id: ジョブID

        Returns:
            ジョブの状態、見つからない場合はNone
        """
        # sacctを使用して完了済みジョブも含めて状態を取得
        cmd = f"sacct -j {job_id} --format=State --noheader --parsable2 | head -1"
        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code == 0 and stdout.strip():
            # "COMPLETED" や "FAILED" など
            state = stdout.strip().split("|")[0].strip()
            if state:
                return state

        # squeueでも確認
        jobs = self.status(job_id=job_id)
        if jobs:
            return jobs[0].state

        return None

    def cancel(self, job_id: str) -> None:
        """ジョブをキャンセル.

        Args:
            job_id: キャンセルするジョブID

        Raises:
            SlurmError: キャンセルに失敗した場合
        """
        stdout, stderr, exit_code = self.ssh.execute(f"scancel {job_id}")

        if exit_code != 0:
            raise SlurmError(f"ジョブのキャンセルに失敗しました: {stderr}")

        console.print(f"[yellow]ジョブ {job_id} をキャンセルしました[/yellow]")

    def get_job_output_file(self, job_id: str) -> str | None:
        """ジョブの標準出力ファイルパスを取得.

        Args:
            job_id: ジョブID

        Returns:
            標準出力ファイルのパス、見つからない場合はNone
        """
        # scontrolでジョブ情報を取得
        cmd = f"scontrol show job {job_id} 2>/dev/null | grep StdOut"
        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code == 0 and stdout.strip():
            # StdOut=/path/to/file の形式
            match = re.search(r"StdOut=(\S+)", stdout)
            if match:
                return match.group(1)

        return None

    def get_job_error_file(self, job_id: str) -> str | None:
        """ジョブの標準エラー出力ファイルパスを取得.

        Args:
            job_id: ジョブID

        Returns:
            標準エラー出力ファイルのパス、見つからない場合はNone
        """
        # scontrolでジョブ情報を取得
        cmd = f"scontrol show job {job_id} 2>/dev/null | grep StdErr"
        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code == 0 and stdout.strip():
            # StdErr=/path/to/file の形式
            match = re.search(r"StdErr=(\S+)", stdout)
            if match:
                return match.group(1)

        return None

    def get_job_error_info(self, job_id: str) -> dict | None:
        """ジョブのエラー情報を取得.

        Args:
            job_id: ジョブID

        Returns:
            エラー情報の辞書、見つからない場合はNone
            - state: ジョブの状態
            - exit_code: 終了コード
            - nodelist: 実行ノード
        """
        # sacctでExitCode、State、NodeListを取得
        cmd = f"sacct -j {job_id} --format=JobID,State,ExitCode,NodeList --noheader --parsable2"
        stdout, stderr, exit_code = self.ssh.execute(cmd)

        if exit_code != 0 or not stdout.strip():
            return None

        # 最初の行（メインジョブ）をパース
        lines = stdout.strip().split("\n")
        if not lines:
            return None

        parts = lines[0].split("|")
        if len(parts) >= 3:
            return {
                "job_id": parts[0],
                "state": parts[1],
                "exit_code": parts[2],
                "nodelist": parts[3] if len(parts) > 3 else "",
            }

        return None

    def get_file_tail(self, file_path: str, max_lines: int = 20) -> list[str]:
        """ファイルの末尾を取得.

        Args:
            file_path: ファイルパス
            max_lines: 取得する最大行数

        Returns:
            ファイルの末尾の行リスト
        """
        cmd = f"tail -n {max_lines} {file_path} 2>/dev/null"
        stdout, _, exit_code = self.ssh.execute(cmd)

        if exit_code != 0:
            return []

        return [line for line in stdout.split("\n") if line.strip()]

    def get_log_tail(
        self, log_path: str, offset: int = 0, max_lines: int = 10
    ) -> tuple[list[str], int]:
        """ログファイルの末尾を取得.

        Args:
            log_path: ログファイルのパス
            offset: 読み取り開始位置（バイト）
            max_lines: 取得する最大行数

        Returns:
            (新しい行のリスト, 新しいオフセット) のタプル
        """
        # ファイルサイズを取得
        size_cmd = f"stat -c %s {log_path} 2>/dev/null || echo 0"
        stdout, _, exit_code = self.ssh.execute(size_cmd)
        if exit_code != 0:
            return [], offset

        try:
            file_size = int(stdout.strip())
        except ValueError:
            return [], offset

        if file_size <= offset:
            return [], offset

        # 新しい部分のみ読み取り
        if offset > 0:
            # 差分読み取り（tail -c +offset で offset バイト目から）
            cmd = f"tail -c +{offset + 1} {log_path} 2>/dev/null | tail -n {max_lines}"
        else:
            cmd = f"tail -n {max_lines} {log_path} 2>/dev/null"

        stdout, _, exit_code = self.ssh.execute(cmd)
        if exit_code != 0:
            return [], offset

        lines = stdout.split("\n")
        return lines, file_size

    def _build_monitor_display(self, monitor: JobMonitor, job_info: JobInfo | None) -> Group:
        """監視表示を構築."""
        state = MonitorState(
            job_id=monitor.job_id,
            state=monitor.state,
            slurm_time=monitor.slurm_time,
            elapsed_time=monitor.elapsed_time,
            log_lines=monitor.log_lines,
        )
        return MonitorDisplayBuilder(state, job_info).build()

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int | None = None,
        callback: Callable[[JobInfo | None, str | None], None] | None = None,
        show_log: bool = True,
        log_poll_interval: int = 5,
    ) -> str:
        """ジョブ完了まで待機.

        Args:
            job_id: 待機するジョブID
            poll_interval: 状態ポーリング間隔 (秒)
            timeout: タイムアウト (秒、None=無制限)
            callback: 状態変化時に呼ばれるコールバック (job_info, state) -> None
            show_log: ログを表示するかどうか
            log_poll_interval: ログポーリング間隔 (秒)

        Returns:
            最終状態 (COMPLETED, FAILED, TIMEOUT, など)

        Raises:
            SlurmError: タイムアウトした場合
        """
        monitor = JobMonitor(job_id=job_id)
        last_state_check = 0.0
        last_log_check = 0.0
        stdout_path: str | None = None
        stderr_path: str | None = None
        cached_job_info: JobInfo | None = None

        def update_display(live: Live) -> str | None:
            """表示を更新し、終了状態があれば返す."""
            nonlocal last_state_check, last_log_check, stdout_path, stderr_path, cached_job_info

            current_time = time.time()
            final_state: str | None = None

            # 状態チェック
            if current_time - last_state_check >= poll_interval or last_state_check == 0:
                last_state_check = current_time

                # タイムアウトチェック
                if timeout is not None:
                    elapsed = current_time - monitor.start_time
                    if elapsed > timeout:
                        raise SlurmError(f"ジョブ {job_id} がタイムアウトしました ({timeout}秒)")

                state = self.get_job_state(job_id)
                jobs = self.status(job_id=job_id)
                cached_job_info = jobs[0] if jobs else None

                if cached_job_info:
                    monitor.slurm_time = cached_job_info.time

                if state != monitor.state:
                    monitor.state = state or "UNKNOWN"
                    if callback:
                        callback(cached_job_info, state)

                    # ログファイルパスを取得（RUNNINGになったら）
                    if state in ("RUNNING", "R") and stdout_path is None and show_log:
                        stdout_path = self.get_job_output_file(job_id)
                        stderr_path = self.get_job_error_file(job_id)
                        # stdout と stderr が同じ場合は stderr を監視しない
                        if stderr_path == stdout_path:
                            stderr_path = None

                # 完了判定
                if state is None or state in ("COMPLETED", "CD"):
                    final_state = "COMPLETED"
                elif state in ("FAILED", "F"):
                    final_state = "FAILED"
                elif state in ("TIMEOUT", "TO"):
                    final_state = "TIMEOUT"
                elif state in ("CANCELLED", "CA"):
                    final_state = "CANCELLED"
                elif state in ("NODE_FAIL", "NF"):
                    final_state = "NODE_FAIL"

            # ログチェック（より頻繁に）
            if show_log and current_time - last_log_check >= log_poll_interval:
                last_log_check = current_time
                # stdout を監視
                if stdout_path:
                    new_lines, new_offset = self.get_log_tail(
                        stdout_path, monitor.last_stdout_offset, max_lines=100
                    )
                    if new_lines:
                        monitor.add_log_lines(new_lines, is_stderr=False)
                        monitor.last_stdout_offset = new_offset
                # stderr を監視
                if stderr_path:
                    new_lines, new_offset = self.get_log_tail(
                        stderr_path, monitor.last_stderr_offset, max_lines=100
                    )
                    if new_lines:
                        monitor.add_log_lines(new_lines, is_stderr=True)
                        monitor.last_stderr_offset = new_offset

            # 毎回表示を更新（経過時間のため）
            live.update(self._build_monitor_display(monitor, cached_job_info))

            return final_state

        # 初期状態を取得
        initial_state = self.get_job_state(job_id)
        jobs = self.status(job_id=job_id)
        cached_job_info = jobs[0] if jobs else None
        monitor.state = initial_state or "UNKNOWN"
        if cached_job_info:
            monitor.slurm_time = cached_job_info.time

        with Live(
            self._build_monitor_display(monitor, cached_job_info),
            console=console,
            refresh_per_second=1,
            transient=False,
        ) as live:
            while True:
                final_state = update_display(live)
                if final_state:
                    # 最終状態を表示
                    if final_state == "COMPLETED":
                        console.print(
                            f"\n[bold green]✓ ジョブ {job_id} が完了しました[/bold green]"
                        )
                    elif final_state in ("FAILED", "TIMEOUT", "NODE_FAIL"):
                        console.print(
                            f"\n[bold red]✗ ジョブ {job_id} が失敗しました ({final_state})[/bold red]"
                        )
                        self._print_error_details(job_id, stdout_path)
                    elif final_state == "CANCELLED":
                        console.print(
                            f"\n[bold yellow]! ジョブ {job_id} がキャンセルされました[/bold yellow]"
                        )
                    return final_state
                time.sleep(1)

    def _print_error_details(self, job_id: str, log_path: str | None) -> None:
        """ジョブ失敗時のエラー詳細を表示.

        Args:
            job_id: ジョブID
            log_path: ログファイルのパス（Noneの場合は取得を試みる）
        """
        # エラー情報を取得
        error_info = self.get_job_error_info(job_id)
        if error_info:
            exit_code = error_info.get("exit_code", "N/A")
            nodelist = error_info.get("nodelist", "")
            console.print(f"[red]  ExitCode: {exit_code}[/red]")
            if nodelist:
                console.print(f"[red]  Node: {nodelist}[/red]")

        # ログファイルパスを取得（未取得の場合）
        stdout_path = log_path or self.get_job_output_file(job_id)
        stderr_path = self.get_job_error_file(job_id)

        # stderrの内容を表示（stdoutと異なる場合のみ）
        if stderr_path and stderr_path != stdout_path:
            stderr_lines = self.get_file_tail(stderr_path, max_lines=20)
            if stderr_lines:
                console.print("\n[bold red]--- stderr ---[/bold red]")
                for line in stderr_lines:
                    console.print(f"[red]{line}[/red]")

        # stdoutの最後の部分を表示
        if stdout_path:
            stdout_lines = self.get_file_tail(stdout_path, max_lines=20)
            if stdout_lines:
                console.print("\n[bold yellow]--- stdout (last 20 lines) ---[/bold yellow]")
                for line in stdout_lines:
                    console.print(f"[dim]{line}[/dim]")

    def print_status_table(self, jobs: list[JobInfo]) -> None:
        """ジョブ状態をテーブル形式で表示.

        Args:
            jobs: 表示するジョブ情報のリスト
        """
        print_job_status_table(jobs)
