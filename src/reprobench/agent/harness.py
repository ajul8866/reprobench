"""Harness adapters for ReproBench agent runs.

Three adapters implement a common :class:`Harness` protocol:

* ``OpenCodeHarness``  - drives the non-interactive ``opencode run --model
  <provider>/<model> --format json`` CLI inside the agent container.
* ``OpenAICompatHarness`` - a minimal OpenAI-compatible chat-completions
  tool loop (tools: bash/read/write/glob/grep/webfetch) used when the
  opencode CLI is unavailable.
* ``ReplayHarness`` - a *test double only*: replays a recorded/fixture
  trace into ``/output`` and ``/trace`` with no network and no model call.
  It is never used for a scored benchmark run (the scorer records
  ``adapter="replay"`` so reports can be labelled fixture-derived).

Every adapter writes:

    <run_dir>/prompt.txt          rendered Fig. 3 prompt
    <run_dir>/output/...          everything the agent produced
    <run_dir>/trace/events.jsonl  trace events (messages + tool calls)
    <run_dir>/meta.json           image digest, tool versions, adapter, cost

Trace event shape (JSONL, one or more events per message)::

    {"type": "message", "role": "assistant",
     "content": "...",
     "tool_calls": [{"name": "bash", "input": {"command": "..."},
                     "output": {"stdout": "...", "exit_code": 0}}],
     "tokens": {"input": 10, "output": 5, "reasoning": 0,
                "cache": {"read": 0, "write": 0}},
     "cost": 0.0012}
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from . import trace as trace_mod
from .budget import BudgetExceeded, CostMonitor, WallClockWatchdog
from .prompt import render_prompt, save_prompt
from ..utils.io import ensure_dir, read_text, write_json, write_text
from ..utils.log import get_logger

LOG = get_logger("agent.harness")

SANDBOX_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TZ",
    "PYTHONPATH",
    "NVD_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
)


@dataclass
class RunConfig:
    """Configuration for a single ``(cve, model, trial)`` run."""

    cve_id: str
    model: str = "glm-5.2"
    trial: int = 1
    adapter: str = "replay"
    runs_dir: Path = Path("results/runs")
    fixture_dir: Optional[Path] = None
    budget_usd: float = 5.0
    max_hours: float = 24.0
    image: str = "reprobench/agent:24.04"
    provider: Optional[str] = None
    judge_model: Optional[str] = None
    opencode_bin: str = "opencode"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        base = f"{self.cve_id}_{self.model}_t{self.trial}".replace("/", "-")
        return base.replace(" ", "_")

    @property
    def provider_model(self) -> str:
        if self.provider:
            return f"{self.provider}/{self.model}"
        if "/" in self.model:
            return self.model
        # Best-effort provider inference for the configured model families.
        lowered = self.model.lower()
        if lowered.startswith(("claude",)):
            return f"anthropic/{self.model}"
        if lowered.startswith(("gpt", "o1", "o3", "o4")):
            return f"openai/{self.model}"
        if lowered.startswith(("glm",)):
            return f"zhipu/{self.model}"
        if lowered.startswith(("mimo",)):
            return f"xiaomi/{self.model}"
        if lowered.startswith(("deepseek",)):
            return f"deepseek/{self.model}"
        return self.model


class Harness(Protocol):  # pragma: no cover - typing only
    name: str

    def run(self, config: RunConfig) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------


def prepare_run_dir(config: RunConfig) -> Dict[str, Path]:
    """Create ``<run_dir>/{output,trace}`` and persist the prompt."""

    run_dir = Path(config.runs_dir) / config.run_id
    output_dir = ensure_dir(run_dir / "output")
    trace_dir = ensure_dir(run_dir / "trace")
    save_prompt(config.cve_id, run_dir)
    return {"run_dir": run_dir, "output_dir": output_dir, "trace_dir": trace_dir}


def write_meta(
    paths: Dict[str, Path],
    config: RunConfig,
    *,
    adapter: str,
    terminated_by: str = "completed",
    cost_usd: float = 0.0,
    wall_seconds: float = 0.0,
    tokens: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``meta.json`` for the run (image digest + tool versions)."""

    meta: Dict[str, Any] = {
        "run_id": config.run_id,
        "cve_id": config.cve_id,
        "model": config.model,
        "trial": config.trial,
        "adapter": adapter,
        "image": config.image,
        "terminated_by": terminated_by,
        "cost_usd": round(cost_usd, 6),
        "wall_seconds": round(wall_seconds, 2),
        "budget_usd": config.budget_usd,
        "max_hours": config.max_hours,
        "fixture_derived": adapter == "replay",
        "prompt_version": "fig3-v1",
        "tool_versions": tool_versions(),
        "tokens": tokens or {},
        "created_at": trace_mod.utc_now(),
    }
    if extra:
        meta.update(extra)
    path = Path(paths["run_dir"]) / "meta.json"
    write_json(path, meta)
    return path


def tool_versions() -> Dict[str, str]:
    """Best-effort tool inventory for the run metadata."""

    versions: Dict[str, str] = {}
    for tool in ("binwalk", "qemu-arm-static", "qemu-system-arm", "ghidra", "opencode"):
        binary = shutil.which(tool)
        if not binary:
            versions[tool] = "absent"
            continue
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            line = (proc.stdout or proc.stderr or "").strip().splitlines()
            versions[tool] = line[0][:120] if line else "present"
        except Exception:  # pragma: no cover - environment dependent
            versions[tool] = "present"
    return versions


def sandbox_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {k: os.environ[k] for k in SANDBOX_ENV_PASSTHROUGH if k in os.environ}
    if extra:
        env.update(extra)
    return env


def _docker_available() -> bool:
    return shutil.which("docker") is not None


# ---------------------------------------------------------------------------
# ReplayHarness (test double only)
# ---------------------------------------------------------------------------


class ReplayHarness:
    """Replays a recorded run directory into ``/output`` and ``/trace``.

    This adapter exists so the scoring pipeline can be exercised offline with
    no model endpoint. It never claims a real rehosting attempt: the fixture
    workspace is copied verbatim and ``meta.json`` records
    ``fixture_derived: true``.
    """

    name = "replay"

    def __init__(self, fixtures_root: Optional[Path] = None) -> None:
        self.fixtures_root = Path(fixtures_root) if fixtures_root else None

    # -- fixture discovery --------------------------------------------------
    def fixture_for(self, config: RunConfig) -> Optional[Path]:
        candidates: List[Path] = []
        if config.fixture_dir:
            root = Path(config.fixture_dir)
            candidates += [
                root / config.run_id,
                root / config.cve_id / config.model / f"trial{config.trial}",
                root / config.cve_id,
                root,
            ]
        if self.fixtures_root:
            root = self.fixtures_root
            candidates += [
                root / config.run_id,
                root / config.cve_id / config.model / f"trial{config.trial}",
                root / config.cve_id,
            ]
        for cand in candidates:
            if cand and Path(cand).is_dir():
                return Path(cand)
        return None

    def run(self, config: RunConfig) -> Dict[str, Any]:
        paths = prepare_run_dir(config)
        fixture = self.fixture_for(config)
        started = time.time()
        if fixture is None:
            LOG.warning("replay: no fixture for %s; emitting empty workspace", config.run_id)
            write_text(
                paths["output_dir"] / "report.md",
                "# Reproduction report\n\nNo replay fixture was available for this run.\n",
            )
            write_text(
                paths["output_dir"] / "plan.md",
                "# Plan\n\nReplay fixture missing; no plan was recorded.\n",
            )
            trace_mod.write_events(
                paths["trace_dir"],
                [
                    trace_mod.message_event(
                        "system",
                        "Replay fixture missing; emitting empty workspace.",
                        tokens={"input": 0, "output": 0},
                        cost=0.0,
                    )
                ],
            )
            write_meta(
                paths,
                config,
                adapter=self.name,
                terminated_by="fixture_missing",
                cost_usd=0.0,
                wall_seconds=time.time() - started,
            )
            return {
                "run_id": config.run_id,
                "run_dir": str(paths["run_dir"]),
                "terminated_by": "fixture_missing",
                "adapter": self.name,
                "fixture": None,
            }

        out_src = fixture / "output"
        if out_src.is_dir():
            _copytree(out_src, paths["output_dir"])
        else:
            _copytree(fixture, paths["output_dir"], skip_names={"trace", "meta.json"})

        trace_src = fixture / "trace"
        if trace_src.is_dir():
            _copytree(trace_src, paths["trace_dir"])
        elif (fixture / "events.jsonl").is_file():
            shutil.copy2(fixture / "events.jsonl", paths["trace_dir"] / "events.jsonl")

        if not (paths["trace_dir"] / "events.jsonl").is_file():
            # Fall back to the run's own prompt so the trace is never empty.
            trace_mod.write_events(
                paths["trace_dir"],
                [
                    trace_mod.message_event("user", read_text(paths["run_dir"] / "prompt.txt", "")),
                    trace_mod.message_event("assistant", "Replayed fixture workspace."),
                ],
            )

        meta_src = fixture / "meta.json"
        tokens: Dict[str, Any] = {}
        cost = 0.0
        if meta_src.is_file():
            try:
                fixture_meta = json.loads(read_text(meta_src, "{}") or "{}")
            except json.JSONDecodeError:
                fixture_meta = {}
            tokens = fixture_meta.get("tokens", {}) or {}
            cost = float(fixture_meta.get("cost_usd", 0.0) or 0.0)

        write_meta(
            paths,
            config,
            adapter=self.name,
            cost_usd=cost,
            wall_seconds=time.time() - started,
            tokens=tokens,
            extra={"fixture": str(fixture)},
        )
        return {
            "run_id": config.run_id,
            "run_dir": str(paths["run_dir"]),
            "terminated_by": "completed",
            "adapter": self.name,
            "fixture": str(fixture),
        }


def _copytree(src: Path, dst: Path, skip_names: Optional[Iterable[str]] = None) -> None:
    skip = set(skip_names or ())
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(Path(src).iterdir()):
        if item.name in skip:
            continue
        target = Path(dst) / item.name
        if item.is_dir():
            _copytree(item, target)
        else:
            shutil.copy2(item, target)


# ---------------------------------------------------------------------------
# OpenCodeHarness
# ---------------------------------------------------------------------------


class OpenCodeHarness:
    """Drives ``opencode run --model <provider>/<model> --format json``.

    When Docker is available and the agent image is built, the CLI runs inside
    a fresh ``ubuntu:24.04`` container (``--rm``) with ``/output`` and
    ``/trace`` bind-mounted. Profiles are *never* mounted. Without Docker the
    CLI is executed directly on the host (development convenience only).
    """

    name = "opencode"

    def __init__(
        self,
        *,
        use_docker: Optional[bool] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        stream_parser: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.use_docker = _docker_available() if use_docker is None else use_docker
        self.timeout = timeout
        self.env = env or {}
        self.stream_parser = stream_parser

    # -- command construction ----------------------------------------------
    def build_command(self, config: RunConfig, paths: Dict[str, Path]) -> List[str]:
        prompt = read_text(paths["run_dir"] / "prompt.txt", "")
        inner = [
            config.opencode_bin,
            "run",
            "--model",
            config.provider_model,
            "--format",
            "json",
            prompt,
        ]
        if not self.use_docker:
            return inner
        docker = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "bridge",
            "-v",
            f"{paths['output_dir']}:/output",
            "-v",
            f"{paths['trace_dir']}:/trace",
            "-w",
            "/output",
        ]
        for key in SANDBOX_ENV_PASSTHROUGH:
            if key in os.environ and key.endswith("_API_KEY"):
                docker += ["-e", key]
        docker += [config.image] + inner
        return docker

    def run(self, config: RunConfig) -> Dict[str, Any]:
        paths = prepare_run_dir(config)
        cmd = self.build_command(config, paths)
        started = time.time()
        monitor = CostMonitor(budget_usd=config.budget_usd)
        watchdog = WallClockWatchdog(max_hours=config.max_hours)
        terminated_by = "completed"
        exit_code: Optional[int] = None
        stderr_tail = ""
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=sandbox_env(self.env),
            )
            exit_code = proc.returncode
            stderr_tail = (proc.stderr or "")[-4000:]
            events = self._parse_stream(proc.stdout or "")
            trace_mod.write_events(paths["trace_dir"], events)
            for event in events:
                monitor.observe(event.get("cost", 0.0))
        except subprocess.TimeoutExpired:
            terminated_by = "watchdog_24h"
        except FileNotFoundError:
            terminated_by = "harness_error"
            stderr_tail = f"{cmd[0]} is not installed"
        except BudgetExceeded:
            terminated_by = "budget_abort"

        if monitor.aborted and terminated_by == "completed":
            terminated_by = "budget_abort"
        if watchdog.expired and terminated_by == "completed":
            terminated_by = "watchdog_24h"
        if exit_code not in (None, 0) and terminated_by == "completed":
            terminated_by = "agent_error"

        if stderr_tail:
            write_text(paths["trace_dir"] / "stderr.log", stderr_tail)

        write_meta(
            paths,
            config,
            adapter=self.name,
            terminated_by=terminated_by,
            cost_usd=monitor.total_cost,
            wall_seconds=time.time() - started,
            tokens=monitor.token_totals(),
            extra={"exit_code": exit_code, "command": cmd[:8]},
        )
        return {
            "run_id": config.run_id,
            "run_dir": str(paths["run_dir"]),
            "terminated_by": terminated_by,
            "adapter": self.name,
            "exit_code": exit_code,
            "cost_usd": round(monitor.total_cost, 6),
        }

    def _parse_stream(self, stdout: str) -> List[Dict[str, Any]]:
        if self.stream_parser is not None:
            return self.stream_parser(stdout)
        return trace_mod.parse_opencode_stream(stdout)


# ---------------------------------------------------------------------------
# OpenAICompatHarness (tool loop fallback)
# ---------------------------------------------------------------------------

TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file (optionally a line range).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write text to a file under /output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetch a URL and return the response body (truncated).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


class OpenAICompatHarness:
    """Minimal OpenAI-compatible tool loop (no vendor SDK required)."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        max_turns: int = 60,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = client
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout

    # -- transport ----------------------------------------------------------
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is not None:  # pragma: no cover - injected test client
            return self.client.chat(payload)
        import httpx  # lazy: optional dependency

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # -- tool implementations ----------------------------------------------
    def execute_tool(self, name: str, args: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        try:
            if name == "bash":
                proc = subprocess.run(  # noqa: S603 - intentional shell for the tool
                    str(args.get("command", "")),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=str(output_dir),
                )
                return {
                    "stdout": (proc.stdout or "")[-4000:],
                    "stderr": (proc.stderr or "")[-2000:],
                    "exit_code": proc.returncode,
                }
            if name == "read":
                path = _resolve_under(Path(args.get("path", "")), output_dir)
                text = read_text(path, "")
                offset = int(args.get("offset") or 0)
                limit = int(args.get("limit") or 400)
                lines = text.splitlines()[offset : offset + limit]
                return {"content": "\n".join(lines), "path": str(path)}
            if name == "write":
                path = _resolve_under(Path(args.get("path", "")), output_dir)
                write_text(path, str(args.get("content", "")))
                return {"path": str(path), "bytes": len(str(args.get("content", "")).encode())}
            if name == "glob":
                pattern = str(args.get("pattern", "*"))
                matches = sorted(str(p.relative_to(output_dir)) for p in output_dir.glob(pattern))
                return {"matches": matches[:200]}
            if name == "grep":
                pattern = str(args.get("pattern", ""))
                root = _resolve_under(Path(args.get("path", ".")), output_dir)
                hits: List[str] = []
                files = [root] if root.is_file() else sorted(root.rglob("*"))
                for fp in files:
                    if not fp.is_file():
                        continue
                    try:
                        text = read_text(fp, "")
                    except OSError:
                        continue
                    for lineno, line in enumerate(text.splitlines(), start=1):
                        if pattern and pattern in line:
                            hits.append(f"{fp.relative_to(output_dir)}:{lineno}:{line[:200]}")
                            if len(hits) >= 200:
                                break
                    if len(hits) >= 200:
                        break
                return {"matches": hits}
            if name == "webfetch":
                import httpx  # lazy: optional dependency

                resp = httpx.get(str(args.get("url", "")), timeout=60.0, follow_redirects=True)
                return {"status": resp.status_code, "body": (resp.text or "")[:8000]}
            return {"error": f"unknown tool {name}"}
        except Exception as exc:  # pragma: no cover - tool failures are data
            return {"error": f"{type(exc).__name__}: {exc}"}

    # -- loop ---------------------------------------------------------------
    def run(self, config: RunConfig) -> Dict[str, Any]:
        paths = prepare_run_dir(config)
        prompt = read_text(paths["run_dir"] / "prompt.txt", render_prompt(config.cve_id))
        output_dir = paths["output_dir"]
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        events: List[Dict[str, Any]] = [trace_mod.message_event("user", prompt)]
        monitor = CostMonitor(budget_usd=config.budget_usd)
        watchdog = WallClockWatchdog(max_hours=config.max_hours)
        terminated_by = "completed"
        started = time.time()

        for turn in range(self.max_turns):
            if config.max_hours is not None and watchdog.expired:
                terminated_by = "watchdog_24h"
                break
            payload = {
                "model": config.model,
                "messages": messages,
                "tools": TOOL_SCHEMA,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
            try:
                response = self._post(payload)
            except Exception as exc:  # pragma: no cover - network path
                terminated_by = "provider_error"
                events.append(trace_mod.message_event("system", f"provider error: {exc}"))
                break

            message = (response.get("choices") or [{}])[0].get("message", {}) or {}
            usage = trace_mod.tokens_from_usage(response.get("usage", {}))
            cost = float(response.get("cost", 0.0) or 0.0)
            monitor.observe(cost, usage)
            tool_calls = message.get("tool_calls") or []
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }
            )
            events.append(
                trace_mod.assistant_event(
                    message.get("content") or "",
                    tool_calls=[],
                    tokens=usage,
                    cost=cost,
                )
            )
            if not tool_calls:
                break
            for call in tool_calls:
                fn = call.get("function", {}) or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {}
                output = self.execute_tool(fn.get("name", ""), args, output_dir)
                record = trace_mod.tool_call(fn.get("name", ""), args, output)
                events.append(
                    trace_mod.assistant_event("", tool_calls=[record], tokens={}, cost=0.0)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"call_{turn}"),
                        "name": fn.get("name", ""),
                        "content": json.dumps(output, default=str)[:8000],
                    }
                )
            if monitor.aborted:
                terminated_by = "budget_abort"
                break

        trace_mod.write_events(paths["trace_dir"], events)
        write_meta(
            paths,
            config,
            adapter=self.name,
            terminated_by=terminated_by,
            cost_usd=monitor.total_cost,
            wall_seconds=time.time() - started,
            tokens=monitor.token_totals(),
            extra={"turns": len(messages)},
        )
        return {
            "run_id": config.run_id,
            "run_dir": str(paths["run_dir"]),
            "terminated_by": terminated_by,
            "adapter": self.name,
            "cost_usd": round(monitor.total_cost, 6),
        }


def _resolve_under(path: Path, root: Path) -> Path:
    """Contain tool file access to the output directory (best effort)."""

    root = Path(root).resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - impossible paths
        return candidate
    try:
        resolved.relative_to(root)
    except ValueError:
        # Allow absolute /output paths from inside the container.
        if str(candidate).startswith("/output"):
            return root / str(candidate).replace("/output", "").lstrip("/")
        return root / resolved.name
    return resolved


ADAPTERS = {
    "replay": ReplayHarness,
    "opencode": OpenCodeHarness,
    "openai_compat": OpenAICompatHarness,
}


def get_harness(adapter: str, **kwargs: Any) -> Any:
    """Instantiate a harness adapter by name."""

    key = (adapter or "replay").strip().lower()
    if key not in ADAPTERS:
        raise ValueError(f"unknown harness adapter: {adapter!r} (have {sorted(ADAPTERS)})")
    return ADAPTERS[key](**kwargs)


__all__ = [
    "ADAPTERS",
    "Harness",
    "OpenAICompatHarness",
    "OpenCodeHarness",
    "ReplayHarness",
    "RunConfig",
    "TOOL_SCHEMA",
    "get_harness",
    "prepare_run_dir",
    "sandbox_env",
    "tool_versions",
    "write_meta",
]
