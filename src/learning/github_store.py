from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


def _runtime_secret(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value:
        return str(value).strip()
    try:
        import streamlit as st

        value = st.secrets.get(key, default)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return default


@dataclass
class GitHubLearningStore:
    repo: str
    token: str
    branch: str = "main"

    @classmethod
    def from_runtime(cls) -> "GitHubLearningStore":
        token = (
            _runtime_secret("LEARNING_GITHUB_TOKEN")
            or _runtime_secret("GITHUB_TOKEN")
            or _runtime_secret("GH_TOKEN")
        )
        repo = _runtime_secret("LEARNING_GITHUB_REPO") or _runtime_secret("GITHUB_REPO") or "marthandan77/XauUsd"
        branch = _runtime_secret("LEARNING_GITHUB_BRANCH") or "main"
        return cls(repo=repo, token=token, branch=branch)

    @property
    def enabled(self) -> bool:
        return bool(self.repo and self.token)

    def status(self) -> dict:
        return {"enabled": self.enabled, "repo": self.repo, "branch": self.branch, "token_present": bool(self.token)}

    def _api_url(self, path: str) -> str:
        quoted = urllib.parse.quote(path.strip("/"))
        return f"https://api.github.com/repos/{self.repo}/contents/{quoted}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "xauusd-learning-pipeline",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, url: str, payload: dict | None = None) -> Any:
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers = {**self._headers(), "Content-Type": "application/json"}
        else:
            body = None
            headers = self._headers()
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"GitHub API {method} {url} failed: {exc.code} {details}") from exc

    def get_content_meta(self, path: str):
        if not self.enabled:
            return None
        url = self._api_url(path) + f"?ref={urllib.parse.quote(self.branch)}"
        return self._request("GET", url)

    def read_text(self, path: str) -> str | None:
        meta = self.get_content_meta(path)
        if not meta or isinstance(meta, list):
            return None
        content = str(meta.get("content", ""))
        if not content:
            return ""
        return base64.b64decode(content).decode("utf-8")

    def read_json(self, path: str) -> dict | None:
        text = self.read_text(path)
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def write_text(self, path: str, text: str, message: str) -> dict:
        if not self.enabled:
            return {"ok": False, "reason": "GitHub learning store is not configured"}
        meta = self.get_content_meta(path)
        payload = {
            "message": message,
            "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            "branch": self.branch,
        }
        if meta and isinstance(meta, dict) and meta.get("sha"):
            payload["sha"] = meta["sha"]
        url = self._api_url(path)
        result = self._request("PUT", url, payload)
        return {"ok": True, "path": path, "commit_sha": (result or {}).get("commit", {}).get("sha")}

    def write_json(self, path: str, data: dict, message: str) -> dict:
        text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        return self.write_text(path, text, message)

    def list_dir(self, path: str) -> list[dict]:
        if not self.enabled:
            return []
        meta = self.get_content_meta(path)
        if isinstance(meta, list):
            return meta
        return []

    def list_json_files(self, root: str, max_files: int = 200) -> list[str]:
        files: list[str] = []
        root_entries = self.list_dir(root)
        for entry in sorted(root_entries, key=lambda item: item.get("name", ""), reverse=True):
            if len(files) >= max_files:
                break
            entry_type = entry.get("type")
            entry_path = entry.get("path")
            if entry_type == "file" and str(entry_path).endswith(".json"):
                files.append(str(entry_path))
            elif entry_type == "dir" and entry_path:
                for child in sorted(self.list_dir(str(entry_path)), key=lambda item: item.get("name", ""), reverse=True):
                    if len(files) >= max_files:
                        break
                    child_path = child.get("path")
                    if child.get("type") == "file" and str(child_path).endswith(".json"):
                        files.append(str(child_path))
        return files
