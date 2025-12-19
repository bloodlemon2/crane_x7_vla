# SPDX-License-Identifier: MIT
# Copyright 2025 nop
"""Jinja2ベースのテンプレート処理.

ジョブテンプレートのプレースホルダ展開を行う。
Jinja2を使用し、条件分岐、ループ、デフォルト値などをサポート。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jinja2 import Environment, StrictUndefined, UndefinedError

from lifter.config import load_env_vars
from lifter.core.console import console


class TemplateError(Exception):
    """テンプレート処理エラー."""


class JobGenerator(Protocol):
    """ジョブスクリプト生成プロトコル."""

    def __call__(self, sweep_id: str, run_number: int) -> str:
        """Sweep IDからジョブスクリプトを生成.

        Args:
            sweep_id: W&B Sweep ID
            run_number: このSweep内での実行番号

        Returns:
            ジョブスクリプトの内容
        """
        ...


@dataclass
class TemplateContext:
    """テンプレート展開コンテキスト.

    Attributes:
        sweep_id: W&B Sweep ID
        run_number: このSweep内での実行番号
        env_vars: 環境変数辞書
    """

    sweep_id: str
    run_number: int
    env_vars: dict[str, str]


class TemplateProcessor:
    """Jinja2テンプレートプロセッサ.

    Jinja2を使用してテンプレートを展開する。
    {{ KEY }} 形式のプレースホルダをサポート。
    従来の {{KEY}} 形式（スペースなし）も自動変換される。

    機能:
    - 条件分岐: {% if VAR %}...{% endif %}
    - ループ: {% for item in items %}...{% endfor %}
    - デフォルト値: {{ VAR | default('value') }}
    - フィルタ: {{ VAR | upper }}, {{ VAR | lower }}
    """

    # 自動的に提供される変数
    BUILTIN_VARS = {"SWEEP_ID", "RUN_NUMBER"}

    def __init__(
        self,
        template_content: str,
        strict: bool = False,
    ):
        """プロセッサを初期化.

        Args:
            template_content: テンプレート内容
            strict: True の場合、未定義変数でエラー
        """
        self.strict = strict
        # 従来の {{VAR}} 形式を {{ VAR }} に変換（後方互換性）
        self.template_content = self._normalize_placeholders(template_content)
        self._env = self._create_jinja_env()
        self._placeholders = self._extract_placeholders()

    def _normalize_placeholders(self, content: str) -> str:
        """従来の {{VAR}} 形式を {{ VAR }} 形式に変換.

        Jinja2は {{ VAR }} 形式を期待するため、
        スペースなしの {{VAR}} を変換する。

        Args:
            content: テンプレート内容

        Returns:
            正規化されたテンプレート
        """
        # {{VAR}} を {{ VAR }} に変換（既にスペースがある場合はスキップ）
        # ただし、Jinja2の制御構文 {%...%} は変換しない
        return re.sub(r"\{\{(\w+)\}\}", r"{{ \1 }}", content)

    def _create_jinja_env(self) -> Environment:
        """Jinja2環境を作成."""
        if self.strict:
            return Environment(undefined=StrictUndefined)
        return Environment()

    def _extract_placeholders(self) -> set[str]:
        """テンプレート内のプレースホルダを抽出.

        Jinja2の変数参照を抽出する。
        """
        # {{ VAR }} または {{ VAR | filter }} 形式を抽出
        pattern = r"\{\{\s*(\w+)(?:\s*\|[^}]*)?\s*\}\}"
        return set(re.findall(pattern, self.template_content))

    @property
    def placeholders(self) -> set[str]:
        """プレースホルダ一覧."""
        return self._placeholders

    def validate(self, env_vars: dict[str, str]) -> list[str]:
        """テンプレートを検証し、未定義の変数を返す.

        Args:
            env_vars: 環境変数辞書

        Returns:
            未定義変数のリスト
        """
        available = set(env_vars.keys()) | self.BUILTIN_VARS
        return list(self._placeholders - available)

    def render(self, context: TemplateContext) -> str:
        """テンプレートをレンダリング.

        Args:
            context: 展開コンテキスト

        Returns:
            展開されたスクリプト

        Raises:
            TemplateError: strict=Trueで未定義変数がある場合
        """
        # コンテキスト変数を構築
        variables: dict[str, Any] = {
            "SWEEP_ID": context.sweep_id,
            "RUN_NUMBER": context.run_number,
            **context.env_vars,
        }

        try:
            template = self._env.from_string(self.template_content)
            return template.render(**variables)
        except UndefinedError as e:
            if self.strict:
                raise TemplateError(f"未定義の変数: {e}") from e
            # strict=Falseの場合は警告のみ
            console.print(f"[yellow]警告: {e}[/yellow]")
            # 未定義変数を空文字列に置換して再試行
            env_lenient = Environment()
            template = env_lenient.from_string(self.template_content)
            return template.render(**variables)


def render_template(
    template_content: str,
    env_vars: dict[str, str] | None = None,
    strict: bool = False,
    **extra_vars: str,
) -> str:
    """テンプレートをレンダリング（簡易関数）.

    Args:
        template_content: テンプレート内容
        env_vars: 環境変数辞書
        strict: 未定義変数でエラーにするか
        **extra_vars: 追加の変数

    Returns:
        レンダリングされた文字列
    """
    all_vars = dict(env_vars or {})
    all_vars.update(extra_vars)

    processor = TemplateProcessor(template_content, strict=strict)
    context = TemplateContext(
        sweep_id="",
        run_number=0,
        env_vars=all_vars,
    )
    return processor.render(context)


def create_template_job_generator(
    template_path: Path,
    env_file: Path | str = ".env",
    strict: bool = False,
    extra_vars: dict[str, str] | None = None,
) -> JobGenerator:
    """テンプレートベースのジョブ生成関数を作成.

    Args:
        template_path: テンプレートファイルパス
        env_file: 環境変数ファイルパス
        strict: 未定義変数でエラーにするか
        extra_vars: 追加の変数（.envより優先）

    Returns:
        ジョブ生成関数

    Raises:
        FileNotFoundError: テンプレートが見つからない場合
        TemplateError: strict=Trueで未定義変数がある場合
    """
    if not template_path.exists():
        raise FileNotFoundError(f"テンプレートファイルが見つかりません: {template_path}")

    template_content = template_path.read_text()

    # .envファイルから環境変数を読み込む
    env_path = Path(env_file)
    if not env_path.exists():
        console.print(f"[yellow]警告: .envファイルが見つかりません: {env_path.absolute()}[/yellow]")

    env_vars = load_env_vars(env_file)

    # 追加変数をマージ（空でない値のみ、.envより優先）
    if extra_vars:
        for key, value in extra_vars.items():
            if value:  # 空でない値のみ追加
                env_vars[key] = value

    console.print(f"[dim].envから {len(env_vars)} 個の変数を読み込みました[/dim]")

    processor = TemplateProcessor(template_content, strict=strict)
    console.print(f"[dim]テンプレート内のプレースホルダ: {processor.placeholders}[/dim]")

    # 事前検証
    undefined = processor.validate(env_vars)
    if undefined:
        msg = f"テンプレートで使用されているが、.envに定義されていない変数: {undefined}"
        if strict:
            raise TemplateError(msg)
        console.print(f"[yellow]警告: {msg}[/yellow]")

    def generator(sweep_id: str, run_number: int) -> str:
        context = TemplateContext(
            sweep_id=sweep_id,
            run_number=run_number,
            env_vars=env_vars,
        )
        return processor.render(context)

    return generator


# 後方互換性のためのエイリアス
def create_custom_job_generator(
    template_path: Path,
    env_file: Path | str = ".env",
    extra_vars: dict[str, str] | None = None,
) -> JobGenerator:
    """カスタムジョブ生成関数を作成（後方互換性のためのエイリアス）.

    新しいコードでは create_template_job_generator() を使用してください。
    """
    return create_template_job_generator(
        template_path, env_file, strict=False, extra_vars=extra_vars
    )
