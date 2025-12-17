# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""ジョブ監視モジュール.

Slurmジョブとローカルジョブのモニタリングのための共通クラス。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from lifter.utils import format_duration_timer


@dataclass
class LogLine:
    """ログ行.

    Attributes:
        text: ログテキスト
        is_stderr: stderrからの出力かどうか
    """

    text: str
    is_stderr: bool = False


@dataclass
class BaseJobMonitor:
    """ジョブ監視の基底クラス.

    Attributes:
        job_id: ジョブID
        start_time: 開始時刻（Unix timestamp）
        log_lines: ログ行のキュー
        state: ジョブの状態
    """

    job_id: str
    start_time: float = field(default_factory=time.time)
    log_lines: deque[LogLine] = field(default_factory=lambda: deque(maxlen=100))
    state: str = "UNKNOWN"

    @property
    def elapsed_time(self) -> str:
        """経過時間を取得.

        Returns:
            "H:MM:SS" または "M:SS" 形式の経過時間
        """
        return format_duration_timer(time.time() - self.start_time)

    def add_log_line(self, text: str, is_stderr: bool = False) -> None:
        """ログ行を追加.

        Args:
            text: ログテキスト
            is_stderr: stderrからの出力かどうか
        """
        stripped = text.rstrip()
        if stripped:
            self.log_lines.append(LogLine(text=stripped, is_stderr=is_stderr))

    def add_log_lines(self, lines: list[str], is_stderr: bool = False) -> None:
        """複数のログ行を追加.

        Args:
            lines: ログテキストのリスト
            is_stderr: stderrからの出力かどうか
        """
        for line in lines:
            self.add_log_line(line, is_stderr)

    def get_recent_lines(self, count: int = 5) -> list[LogLine]:
        """最新のログ行を取得.

        Args:
            count: 取得する行数

        Returns:
            最新のログ行リスト
        """
        return list(self.log_lines)[-count:]

    def get_recent_text(self, count: int = 5) -> list[str]:
        """最新のログテキストを取得.

        Args:
            count: 取得する行数

        Returns:
            最新のログテキストリスト
        """
        return [line.text for line in self.get_recent_lines(count)]
