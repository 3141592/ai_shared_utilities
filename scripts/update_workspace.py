#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


SRC_ROOT = Path.home() / "src"
WORKSPACE_FILE = SRC_ROOT / "ai_lab.code-workspace"

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
}

INCLUDED_REPOS = [
    "ai_surgery",
    "deep_learning_with_python",
    "ARENA_3.0",
    "build_a_llm_from_scratch",
    "ai_shared_data",
]

def is_repo_dir(path: Path) -> bool:
    return path.is_dir() and (path / ".git").is_dir()


def repo_paths() -> list[Path]:
    repos = []
    for name in INCLUDED_REPOS:
        path = SRC_ROOT / name
        if not path.exists():
            print(f"Warning: repo not found: {name}")
            continue
        if not (path / ".git").is_dir():
            print(f"Warning: not a git repo: {name}")
            continue
        repos.append(path)
    return sorted(repos, key=lambda p: p.name.lower())


def load_workspace(path: Path) -> dict:
    if not path.exists():
        return {"folders": [], "settings": {}}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def relpath_from_src(path: Path) -> str:
    return path.relative_to(SRC_ROOT).as_posix()


def update_workspace() -> None:
    workspace = load_workspace(WORKSPACE_FILE)
    existing_settings = workspace.get("settings", {})
    existing_extensions = workspace.get("extensions")
    existing_launch = workspace.get("launch")
    existing_tasks = workspace.get("tasks")

    folders = [{"path": relpath_from_src(repo)} for repo in repo_paths()]
    workspace["folders"] = folders
    workspace["settings"] = existing_settings

    if existing_extensions is not None:
        workspace["extensions"] = existing_extensions
    if existing_launch is not None:
        workspace["launch"] = existing_launch
    if existing_tasks is not None:
        workspace["tasks"] = existing_tasks

    with WORKSPACE_FILE.open("w", encoding="utf-8") as f:
        json.dump(workspace, f, indent=2)
        f.write("\n")

    print(f"Updated {WORKSPACE_FILE}")
    print(f"Repo count: {len(folders)}")
    for folder in folders:
        print(f"  {folder['path']}")


if __name__ == "__main__":
    update_workspace()
