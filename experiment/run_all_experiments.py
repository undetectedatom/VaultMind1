"""
Final experiment runner for Python 3.12 Chinese docs + MDN English docs.

Recommended location:
    experiment/run_all_experiments.py

Common usage:
    python experiment/run_all_experiments.py --check
    python experiment/run_all_experiments.py --prepare-only --reset-documents
    python experiment/run_all_experiments.py --all --skip-document-prepare

Important:
- This script does not set HTTP request timeout values.
- This script waits indefinitely for server startup, document embedding, and API responses.
- Rate control is done only through sleep intervals and batch size.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAG_CONFIG_PATH = PROJECT_ROOT / "rag_config.json"
BACKUP_DIR = SCRIPT_DIR / "data" / "results" / "config_backups"
RESULT_DIR = SCRIPT_DIR / "data" / "results"
LOG_DIR = RESULT_DIR / "server_logs"
SUMMARY_DIR = RESULT_DIR / "summaries"

API_BASE_URL = "http://localhost:8000/api"
SERVER_ROOT_URL = "http://localhost:8000"

DEFAULT_SERVER_CMD = [
    sys.executable,
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]

# These are rate-control settings, not timeout settings.
CHAT_REQUEST_INTERVAL_SEC = 3.0
UPLOAD_BATCH_SIZE = 3
UPLOAD_REQUEST_INTERVAL_SEC = 3.0
DOCUMENT_POLL_INTERVAL_SEC = 5

LIGHT_GENERATION_MODEL = {"provider": "doubao", "model": "deepseek-v3.2"}
STRONG_GENERATION_MODEL = {"provider": "doubao", "model": "doubao-seed-2.0-pro"}

CLUSTERS: dict[str, dict[str, Any]] = {
    "python_zh_312_html": {
        "display_name": "Python 3.12 中文官方文档",
        "username": "test_python_zh_312_html",
        "password": "pwd_python_zh_312_html_123",
        "email": "python_zh_312_html@test.com",
        "data_dir": SCRIPT_DIR / "data" / "clusters" / "python_zh_312_html",
        "allowed_extensions": [".html"],
    },
    "mdn_web_docs_en_md": {
        "display_name": "MDN Web Docs 英文技术文档",
        "username": "test_mdn_web_docs_en_md",
        "password": "pwd_mdn_web_docs_en_md_123",
        "email": "mdn_web_docs_en_md@test.com",
        "data_dir": SCRIPT_DIR / "data" / "clusters" / "mdn_web_docs_en_md",
        "allowed_extensions": [".md"],
    },
}

MAIN_EVAL_PATH = SCRIPT_DIR / "data" / "eval" / "main_eval.json"
REJECTION_EVAL_PATH = SCRIPT_DIR / "data" / "eval" / "rejection_eval.json"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "exp01_basic_rag_light_top4": {
        "title": "基础RAG轻量模型基线",
        "paper_section": "4.3.2，4.3.3，4.3.4，4.3.5",
        "purpose": "关闭HyDE与多模型路由，固定使用轻量生成模型，作为基础对照组。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp01_basic_rag_light_top4.csv",
        "rag": {
            "hyde_enabled": False,
            "routing_enabled": False,
            "generation": [LIGHT_GENERATION_MODEL],
            "top_k": 4,
        },
    },
    "exp02_hyde_rag_light_top4": {
        "title": "HyDE增强检索实验",
        "paper_section": "4.3.3，4.3.5",
        "purpose": "仅开启HyDE并固定轻量模型，用于观察HyDE对检索命中和引用命中的影响。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp02_hyde_rag_light_top4.csv",
        "rag": {
            "hyde_enabled": True,
            "routing_enabled": False,
            "generation": [LIGHT_GENERATION_MODEL],
            "top_k": 4,
        },
    },
    "exp03_no_hyde_routing_top4": {
        "title": "无HyDE多模型路由实验",
        "paper_section": "4.3.4，4.3.5",
        "purpose": "关闭HyDE但开启多模型路由，用于判断路由策略在无HyDE输入时的表现。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp03_no_hyde_routing_top4.csv",
        "rag": {
            "hyde_enabled": False,
            "routing_enabled": True,
            "generation": [LIGHT_GENERATION_MODEL, STRONG_GENERATION_MODEL],
            "top_k": 4,
        },
    },
    "exp04_full_routing_top4": {
        "title": "完整系统实验",
        "paper_section": "4.3.3，4.3.4，4.3.5",
        "purpose": "同时开启HyDE与多模型路由，作为本文提出方案的完整形态。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp04_full_routing_top4.csv",
        "rag": {
            "hyde_enabled": True,
            "routing_enabled": True,
            "generation": [LIGHT_GENERATION_MODEL, STRONG_GENERATION_MODEL],
            "top_k": 4,
        },
    },
    "exp05_fixed_strong_top4": {
        "title": "高能力固定模型对照实验",
        "paper_section": "4.3.2，4.3.4",
        "purpose": "开启HyDE但关闭路由，固定使用高能力模型，用于与动态路由方案比较生成效果和token消耗。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp05_fixed_strong_top4.csv",
        "rag": {
            "hyde_enabled": True,
            "routing_enabled": False,
            "generation": [STRONG_GENERATION_MODEL],
            "top_k": 4,
        },
    },
    "exp06_full_routing_top2": {
        "title": "Top-k为2的检索敏感性实验",
        "paper_section": "4.3.3",
        "purpose": "在完整系统下减少候选片段数量，观察较小Top-k对检索和生成的影响。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp06_full_routing_top2.csv",
        "rag": {
            "hyde_enabled": True,
            "routing_enabled": True,
            "generation": [LIGHT_GENERATION_MODEL, STRONG_GENERATION_MODEL],
            "top_k": 2,
        },
    },
    "exp07_full_routing_top6": {
        "title": "Top-k为6的检索敏感性实验",
        "paper_section": "4.3.3",
        "purpose": "在完整系统下增加候选片段数量，观察较大Top-k对召回、噪声、时延和token消耗的影响。",
        "dataset_path": MAIN_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp07_full_routing_top6.csv",
        "rag": {
            "hyde_enabled": True,
            "routing_enabled": True,
            "generation": [LIGHT_GENERATION_MODEL, STRONG_GENERATION_MODEL],
            "top_k": 6,
        },
    },
    "exp08_rejection_full_routing_top4": {
        "title": "越界问题拒答实验",
        "paper_section": "4.3.4，4.3.5",
        "purpose": "使用越界问题集检验系统是否会在证据不足时避免强行引用文档。",
        "dataset_path": REJECTION_EVAL_PATH,
        "output_csv_path": RESULT_DIR / "exp08_rejection_full_routing_top4.csv",
        "rag": {
            "hyde_enabled": True,
            "routing_enabled": True,
            "generation": [LIGHT_GENERATION_MODEL, STRONG_GENERATION_MODEL],
            "top_k": 4,
        },
    },
}

DEFAULT_RAG_CONFIG = {
    "providers": {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "models": {
                "gpt-4o-mini": {"context_window": 128000},
                "text-embedding-3-small": {"dimension": 1536},
            },
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "models": {
                "llama3:8b": {"context_window": 8192},
                "mistral": {"context_window": 8192},
            },
        },
        "openai_doubao_embedding": {
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "models": {
                "doubao-embedding-vision": {"dimension": 1024},
            },
        },
        "doubao": {
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "models": {
                "doubao-seed-2.0-pro": {"context_window": 128000},
                "deepseek-v3.2": {"context_window": 128000},
                "glm-4.7": {"context_window": 128000},
                "kimi-2.6": {"context_window": 128000},
                "doubao-embedding-vision": {"dimension": 2048},
            },
        },
    },
    "activities": {
        "embedding": {"provider": "doubao", "model": "doubao-embedding-vision"},
        "router": {"provider": "doubao", "model": "deepseek-v3.2"},
        "generation": [LIGHT_GENERATION_MODEL, STRONG_GENERATION_MODEL],
    },
    "hyde": {"enabled": True, "temperature": 0.0},
    "routing": {"enabled": True, "confidence_threshold": 0.5},
    "retrieval": {"top_k": 4},
}

SUMMARY_METRICS = [
    "latency_sec",
    "retrieval_context_available",
    "retrieval_document_hit",
    "retrieval_mrr",
    "citation_hit",
    "citation_mrr",
    "citation_decision_accuracy",
    "rejection_accuracy",
    "answer_length",
    "source_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


@dataclass
class ServerHandle:
    process: subprocess.Popen | None = None
    log_file_handle: Any | None = None


class RagConfigManager:
    def __init__(self, rag_config_path: Path = RAG_CONFIG_PATH):
        self.rag_config_path = rag_config_path
        self.original_config: dict[str, Any] | None = None
        self.backup_path: Path | None = None

    def load_base_config(self) -> dict[str, Any]:
        if self.rag_config_path.exists():
            with open(self.rag_config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return json.loads(json.dumps(DEFAULT_RAG_CONFIG))

    def backup_once(self) -> None:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if self.original_config is None:
            self.original_config = self.load_base_config()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.backup_path = BACKUP_DIR / f"rag_config_original_{timestamp}.json"
            with open(self.backup_path, "w", encoding="utf-8") as file:
                json.dump(self.original_config, file, ensure_ascii=False, indent=2)
            print(f"Backed up original rag_config.json to {self.backup_path}")

    def build_config_for_experiment(self, experiment_name: str) -> dict[str, Any]:
        base_config = self.load_base_config()
        experiment = EXPERIMENTS[experiment_name]
        rag = experiment["rag"]

        base_config.setdefault("activities", {})
        base_config.setdefault("hyde", {})
        base_config.setdefault("routing", {})
        base_config.setdefault("retrieval", {})

        base_config["activities"]["generation"] = rag["generation"]
        base_config["hyde"]["enabled"] = rag["hyde_enabled"]
        base_config["hyde"].setdefault("temperature", 0.0)
        base_config["routing"]["enabled"] = rag["routing_enabled"]
        base_config["routing"].setdefault("confidence_threshold", 0.5)
        base_config["retrieval"]["top_k"] = rag["top_k"]

        return base_config

    def write_experiment_config(self, experiment_name: str) -> Path:
        self.backup_once()
        config = self.build_config_for_experiment(experiment_name)

        with open(self.rag_config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        snapshot_dir = RESULT_DIR / "rag_config_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{experiment_name}.rag_config.json"
        with open(snapshot_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        print(f"Wrote rag_config.json for {experiment_name}")
        return snapshot_path

    def restore_original_config(self) -> None:
        if self.original_config is None:
            return
        with open(self.rag_config_path, "w", encoding="utf-8") as file:
            json.dump(self.original_config, file, ensure_ascii=False, indent=2)
        print("Restored original rag_config.json")


class ServerManager:
    def __init__(
        self, server_cmd: list[str] | None = None, use_running_server: bool = False
    ):
        self.server_cmd = server_cmd or DEFAULT_SERVER_CMD
        self.use_running_server = use_running_server
        self.handle = ServerHandle()

    def start(self, experiment_name: str) -> None:
        if self.use_running_server:
            self.wait_for_existing_server()
            print("Using already running FastAPI server.")
            return

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{experiment_name}.server.log"
        log_file = open(log_path, "w", encoding="utf-8")

        print(f"Starting FastAPI server for {experiment_name}.")
        print(f"Server log: {log_path}")

        kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "text": True,
        }

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid

        process = subprocess.Popen(self.server_cmd, **kwargs)
        self.handle = ServerHandle(process=process, log_file_handle=log_file)
        self.wait_until_ready(process, log_path)

    def wait_for_existing_server(self) -> None:
        health_url = f"{SERVER_ROOT_URL}/openapi.json"
        print(f"Waiting indefinitely for existing FastAPI server: {health_url}")

        while True:
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

    def wait_until_ready(self, process: subprocess.Popen, log_path: Path) -> None:
        health_url = f"{SERVER_ROOT_URL}/openapi.json"
        print(f"Waiting indefinitely for FastAPI server to start. See log: {log_path}")

        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"FastAPI server exited early with code {process.returncode}. See log: {log_path}"
                )

            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    print("FastAPI server is ready.")
                    return
            except requests.RequestException:
                pass

            time.sleep(1)

    def stop(self) -> None:
        if self.use_running_server:
            return

        process = self.handle.process
        if process is None:
            return

        if process.poll() is None:
            print("Stopping FastAPI server.")
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()
            except Exception:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()

        if self.handle.log_file_handle is not None:
            self.handle.log_file_handle.close()

        self.handle = ServerHandle()


class ClusterManager:
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url.rstrip("/")
        self.tokens: dict[str, str] = {}

    def register_user_if_needed(self, cluster_name: str) -> None:
        cluster = CLUSTERS[cluster_name]
        payload = {
            "username": cluster["username"],
            "password": cluster["password"],
            "email": cluster["email"],
        }
        response = requests.post(
            f"{self.api_base_url}/v1/users/register",
            json=payload,
        )
        if response.status_code == 201:
            print(f"Registered user for cluster: {cluster_name}")
            return
        if response.status_code == 400:
            print(f"User already exists for cluster: {cluster_name}")
            return
        response.raise_for_status()

    def login(self, cluster_name: str) -> str:
        if cluster_name in self.tokens:
            return self.tokens[cluster_name]

        cluster = CLUSTERS[cluster_name]
        response = requests.post(
            f"{self.api_base_url}/auth/token",
            data={"username": cluster["username"], "password": cluster["password"]},
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        self.tokens[cluster_name] = token
        return token

    def list_documents(self, token: str) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.api_base_url}/v1/documents",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def document_status(doc: dict[str, Any]) -> str:
        return str(
            doc.get("upload_status", "") or doc.get("status", "") or "unknown"
        ).lower()

    def delete_document_by_id(self, token: str, doc_id: str) -> None:
        response = requests.delete(
            f"{self.api_base_url}/v1/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()

    def delete_all_documents(self, cluster_name: str) -> None:
        token = self.login(cluster_name)
        documents = self.list_documents(token)

        for document in tqdm(documents, desc=f"delete_all:{cluster_name}"):
            doc_id = document.get("id")
            if not doc_id:
                continue
            self.delete_document_by_id(token, str(doc_id))
            time.sleep(0.05)

        print(
            f"Deleted {len(documents)} existing documents from cluster: {cluster_name}"
        )

    def delete_non_completed_local_documents(
        self,
        token: str,
        local_filenames: set[str],
        cluster_name: str,
    ) -> None:
        documents = self.list_documents(token)
        stale_docs = [
            doc
            for doc in documents
            if doc.get("filename") in local_filenames
            and self.document_status(doc) != "completed"
        ]

        if not stale_docs:
            return

        print(
            f"Cluster {cluster_name}: found {len(stale_docs)} non-completed local document records. "
            "They will be deleted and reuploaded."
        )

        for document in tqdm(stale_docs, desc=f"delete_stale:{cluster_name}"):
            doc_id = document.get("id")
            if not doc_id:
                continue
            self.delete_document_by_id(token, str(doc_id))
            time.sleep(0.05)

    def iter_local_files(self, cluster_name: str) -> list[Path]:
        cluster = CLUSTERS[cluster_name]
        data_dir = Path(cluster["data_dir"])
        if not data_dir.exists():
            raise FileNotFoundError(f"Cluster data directory not found: {data_dir}")

        allowed = {ext.lower() for ext in cluster.get("allowed_extensions", [])}
        files = [
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        ]
        files = sorted(files, key=lambda path: path.name)

        if not files:
            raise FileNotFoundError(
                f"No files with extensions {sorted(allowed)} found in {data_dir}"
            )

        return files

    def upload_file_batch(self, token: str, file_batch: list[Path]) -> None:
        opened_files = []
        files_payload = []

        try:
            for file_path in file_batch:
                file_obj = open(file_path, "rb")
                opened_files.append(file_obj)
                mime_type, _ = mimetypes.guess_type(str(file_path))
                files_payload.append(
                    (
                        "files",
                        (
                            file_path.name,
                            file_obj,
                            mime_type or "application/octet-stream",
                        ),
                    )
                )

            response = requests.post(
                f"{self.api_base_url}/v1/documents/upload",
                headers={"Authorization": f"Bearer {token}"},
                files=files_payload,
            )
            response.raise_for_status()

            payload = response.json()
            failed = payload.get("failed_files") or []
            if failed:
                raise RuntimeError(f"Upload API reported failed files: {failed}")

        finally:
            for file_obj in opened_files:
                file_obj.close()

    @staticmethod
    def summarize_statuses(documents: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in documents:
            status = ClusterManager.document_status(doc)
            counts[status] = counts.get(status, 0) + 1
        return dict(sorted(counts.items()))

    def wait_for_filenames_ready(
        self,
        token: str,
        filenames: set[str],
        cluster_name: str,
    ) -> None:
        if not filenames:
            return

        last_print = 0.0
        print(
            f"Waiting indefinitely for {len(filenames)} document embeddings: {cluster_name}"
        )

        while True:
            documents = self.list_documents(token)
            target_docs = [doc for doc in documents if doc.get("filename") in filenames]

            status_by_name: dict[str, list[str]] = {}
            for doc in target_docs:
                filename = str(doc.get("filename", ""))
                status_by_name.setdefault(filename, []).append(
                    self.document_status(doc)
                )

            completed = [
                filename
                for filename, statuses in status_by_name.items()
                if "completed" in statuses
            ]
            failed = [
                filename
                for filename, statuses in status_by_name.items()
                if "failed" in statuses
            ]
            missing = sorted(filenames - set(status_by_name))

            now = time.time()
            if now - last_print > 15:
                processing_count = sum(
                    1
                    for statuses in status_by_name.values()
                    if any(status in {"pending", "processing"} for status in statuses)
                )
                print(
                    f"Embedding status for {cluster_name}: "
                    f"completed={len(completed)}, failed={len(failed)}, "
                    f"processing_or_pending={processing_count}, missing={len(missing)}"
                )
                last_print = now

            if failed:
                raise RuntimeError(
                    f"{len(failed)} documents failed to embed for {cluster_name}. "
                    f"Sample failed files: {failed[:10]}"
                )

            if len(completed) >= len(filenames):
                print(
                    f"All target documents completed for {cluster_name}: {len(completed)}/{len(filenames)}"
                )
                return

            time.sleep(DOCUMENT_POLL_INTERVAL_SEC)

    def wait_until_documents_ready(
        self,
        token: str,
        expected_filenames: set[str],
        cluster_name: str,
    ) -> None:
        self.wait_for_filenames_ready(
            token=token,
            filenames=expected_filenames,
            cluster_name=cluster_name,
        )

    def prepare_cluster(self, cluster_name: str, reset_documents: bool = False) -> None:
        print(f"Preparing cluster: {cluster_name}")
        self.register_user_if_needed(cluster_name)
        token = self.login(cluster_name)

        local_files = self.iter_local_files(cluster_name)
        local_filenames = {path.name for path in local_files}

        if reset_documents:
            self.delete_all_documents(cluster_name)
        else:
            self.delete_non_completed_local_documents(
                token=token,
                local_filenames=local_filenames,
                cluster_name=cluster_name,
            )

        existing_docs = self.list_documents(token)
        completed_names = {
            str(doc.get("filename", ""))
            for doc in existing_docs
            if self.document_status(doc) == "completed"
        }

        files_to_upload = [
            path for path in local_files if path.name not in completed_names
        ]

        print(
            f"Cluster {cluster_name}: local_files={len(local_files)}, "
            f"completed_documents={len(completed_names & local_filenames)}, "
            f"need_upload={len(files_to_upload)}"
        )

        for start in range(0, len(files_to_upload), UPLOAD_BATCH_SIZE):
            batch = files_to_upload[start : start + UPLOAD_BATCH_SIZE]
            batch_names = {path.name for path in batch}

            self.upload_file_batch(token, batch)
            print(
                f"Uploaded {start + len(batch)}/{len(files_to_upload)} files to {cluster_name}"
            )

            # Sequentially wait for each uploaded batch to finish before uploading the next one.
            # This avoids creating many concurrent background embedding jobs under strict rate limits.
            self.wait_for_filenames_ready(
                token=token,
                filenames=batch_names,
                cluster_name=cluster_name,
            )

            time.sleep(UPLOAD_REQUEST_INTERVAL_SEC)

        self.wait_until_documents_ready(
            token=token,
            expected_filenames=local_filenames,
            cluster_name=cluster_name,
        )

    def prepare_all_clusters(self, reset_documents: bool = False) -> None:
        for cluster_name in CLUSTERS.keys():
            self.prepare_cluster(cluster_name, reset_documents=reset_documents)


class EvaluationRunner:
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url.rstrip("/")
        self.tokens: dict[str, str] = {}
        self.retrieval_metric_warning_printed = False

    def login(self, cluster_name: str) -> str:
        if cluster_name in self.tokens:
            return self.tokens[cluster_name]

        cluster = CLUSTERS[cluster_name]
        response = requests.post(
            f"{self.api_base_url}/auth/token",
            data={"username": cluster["username"], "password": cluster["password"]},
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        self.tokens[cluster_name] = token
        return token

    @staticmethod
    def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {dataset_path}\n"
                "Run experiment/generate_eval_qa.py before running experiments."
            )

        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError(f"Evaluation dataset must be a JSON array: {dataset_path}")

        return dataset

    @staticmethod
    def validate_dataset_items(
        dataset: list[dict[str, Any]], dataset_path: Path
    ) -> None:
        required = {
            "id",
            "cluster",
            "question_type",
            "query",
            "expected_doc_filename",
            "is_out_of_bounds",
            "reference_answer",
        }

        seen_ids: set[str] = set()

        for item in dataset:
            item_id = str(item.get("id", ""))
            missing = required - set(item)
            if missing:
                raise ValueError(
                    f"{dataset_path}: item {item_id} missing keys {sorted(missing)}"
                )

            if item_id in seen_ids:
                raise ValueError(f"{dataset_path}: duplicate item id {item_id}")
            seen_ids.add(item_id)

            if "answer_keywords" in item:
                raise ValueError(
                    f"{dataset_path}: item {item_id} still contains answer_keywords. "
                    "Regenerate the evaluation set with the new script."
                )

            cluster_name = item.get("cluster")
            if cluster_name not in CLUSTERS:
                raise ValueError(
                    f"{dataset_path}: unknown cluster in item {item_id}: {cluster_name}"
                )

            is_out = bool(item.get("is_out_of_bounds", False))
            expected = str(item.get("expected_doc_filename", "")).strip()

            if is_out and expected:
                raise ValueError(
                    f"{dataset_path}: rejection item {item_id} has expected filename"
                )

            if not is_out and not expected:
                raise ValueError(
                    f"{dataset_path}: normal item {item_id} lacks expected filename"
                )

    @staticmethod
    def normalize_expected_filenames(item: dict[str, Any]) -> list[str]:
        names = item.get("expected_doc_filenames")
        if isinstance(names, list):
            return [str(name).strip() for name in names if str(name).strip()]

        name = str(item.get("expected_doc_filename", "")).strip()
        return [name] if name else []

    @staticmethod
    def source_filenames(sources: list[dict[str, Any]]) -> list[str]:
        filenames: list[str] = []

        for source in sources:
            if not isinstance(source, dict):
                continue

            filename = str(source.get("filename", "")).strip()
            if filename:
                filenames.append(filename)

        return filenames

    @staticmethod
    def unique_keep_order(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            value = str(value).strip()
            if not value or value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result

    def extract_retrieved_filenames_from_meta(self, meta: dict[str, Any]) -> list[str]:
        direct_keys = [
            "retrieved_filenames",
            "retrieved_source_filenames",
            "raw_retrieved_filenames",
            "context_filenames",
        ]

        for key in direct_keys:
            value = meta.get(key)
            if isinstance(value, list):
                filenames = [str(item).strip() for item in value if str(item).strip()]
                if filenames:
                    return self.unique_keep_order(filenames)

        list_dict_keys = [
            "retrieved_sources",
            "retrieved_docs",
            "retrieval_sources",
            "context_sources",
        ]

        for key in list_dict_keys:
            value = meta.get(key)
            if isinstance(value, list):
                filenames = []
                for item in value:
                    if isinstance(item, dict):
                        filename = str(item.get("filename", "")).strip()
                        if filename:
                            filenames.append(filename)

                if filenames:
                    return self.unique_keep_order(filenames)

        if not self.retrieval_metric_warning_printed:
            print(
                "WARNING: response meta does not expose raw retrieved filenames. "
                "retrieval_document_hit and retrieval_mrr will be empty. "
                "Add meta['retrieved_filenames'] in app/services/chat_service.py after retrieval."
            )
            self.retrieval_metric_warning_printed = True

        return []

    @staticmethod
    def evaluate_hit_and_mrr(
        candidate_filenames: list[str],
        expected_filenames: list[str],
    ) -> tuple[float | None, float | None]:
        if not expected_filenames:
            return None, None

        if not candidate_filenames:
            return None, None

        expected_set = set(expected_filenames)
        hit = 1.0 if expected_set.intersection(candidate_filenames) else 0.0

        mrr = 0.0
        for index, filename in enumerate(candidate_filenames):
            if filename in expected_set:
                mrr = 1.0 / (index + 1)
                break

        return hit, mrr

    @staticmethod
    def evaluate_citation_decision(
        source_filenames: list[str],
        expected_filenames: list[str],
        is_out_of_bounds: bool,
    ) -> tuple[float | None, float | None, float]:
        if is_out_of_bounds:
            accuracy = 1.0 if not source_filenames else 0.0
            return None, None, accuracy

        citation_hit, citation_mrr = EvaluationRunner.evaluate_hit_and_mrr(
            source_filenames,
            expected_filenames,
        )
        return citation_hit, citation_mrr, float(citation_hit or 0.0)

    @staticmethod
    def evaluate_rejection_accuracy(
        source_filenames: list[str],
        is_out_of_bounds: bool,
    ) -> float | None:
        if not is_out_of_bounds:
            return None

        return 1.0 if not source_filenames else 0.0

    @staticmethod
    def safe_number(value: Any) -> float:
        try:
            if value is None or value == "":
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def call_chat(
        self, cluster_name: str, query: str
    ) -> tuple[dict[str, Any], str, float]:
        token = self.login(cluster_name)
        headers = {"Authorization": f"Bearer {token}"}

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_base_url}/v1/chat",
                headers=headers,
                json={"query": query},
            )
            response.raise_for_status()
            return response.json(), "", time.time() - start_time
        except Exception as error:
            return (
                {"answer": "", "sources": [], "meta": {}},
                repr(error),
                time.time() - start_time,
            )

    def run_experiment(self, experiment_name: str) -> Path:
        experiment = EXPERIMENTS[experiment_name]
        dataset_path = Path(experiment["dataset_path"])
        output_csv_path = Path(experiment["output_csv_path"])
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = self.load_dataset(dataset_path)
        self.validate_dataset_items(dataset, dataset_path)

        results: list[dict[str, Any]] = []

        for item in tqdm(dataset, desc=experiment_name):
            cluster_name = str(item.get("cluster", ""))
            response_data, request_error, latency = self.call_chat(
                cluster_name,
                str(item["query"]),
            )

            answer = response_data.get("answer", "")
            sources = response_data.get("sources", []) or []
            meta = response_data.get("meta") or response_data.get("metadata", {}) or {}

            if not isinstance(sources, list):
                sources = []

            if not isinstance(meta, dict):
                meta = {}

            expected_filenames = self.normalize_expected_filenames(item)
            is_out_of_bounds = bool(item.get("is_out_of_bounds", False))

            retrieved_filenames = self.extract_retrieved_filenames_from_meta(meta)
            source_filenames = self.source_filenames(sources)

            retrieval_context_available = 1.0 if retrieved_filenames else 0.0

            if is_out_of_bounds:
                retrieval_document_hit = None
                retrieval_mrr = None
            else:
                retrieval_document_hit, retrieval_mrr = self.evaluate_hit_and_mrr(
                    retrieved_filenames,
                    expected_filenames,
                )

            citation_hit, citation_mrr, citation_decision_accuracy = (
                self.evaluate_citation_decision(
                    source_filenames=source_filenames,
                    expected_filenames=expected_filenames,
                    is_out_of_bounds=is_out_of_bounds,
                )
            )
            rejection_accuracy = self.evaluate_rejection_accuracy(
                source_filenames, is_out_of_bounds
            )

            selected_model = str(meta.get("selected_model", "unknown"))
            prompt_tokens = self.safe_number(meta.get("prompt_tokens", 0))
            completion_tokens = self.safe_number(meta.get("completion_tokens", 0))
            total_tokens = prompt_tokens + completion_tokens

            results.append(
                {
                    "experiment": experiment_name,
                    "experiment_title": experiment["title"],
                    "id": item.get("id", ""),
                    "cluster": cluster_name,
                    "cluster_display_name": CLUSTERS[cluster_name]["display_name"],
                    "question_type": item.get("question_type", ""),
                    "query": item.get("query", ""),
                    "reference_answer": item.get("reference_answer", ""),
                    "evidence_hint": item.get("evidence_hint", ""),
                    "expected_doc_filenames": ";".join(expected_filenames),
                    "is_out_of_bounds": is_out_of_bounds,
                    "latency_sec": round(latency, 3),
                    "retrieval_context_available": retrieval_context_available,
                    "retrieval_document_hit": retrieval_document_hit,
                    "retrieval_mrr": retrieval_mrr,
                    "citation_hit": citation_hit,
                    "citation_mrr": citation_mrr,
                    "citation_decision_accuracy": citation_decision_accuracy,
                    "rejection_accuracy": rejection_accuracy,
                    "answer_length": len(str(answer)),
                    "source_count": len(source_filenames),
                    "selected_model": selected_model,
                    "llm_complexity": meta.get("llm_complexity_score", None),
                    "retrieval_confidence": meta.get("retrieval_confidence", None),
                    "vector_retrieved_count": meta.get("vector_retrieved_count", None),
                    "bm25_retrieved_count": meta.get("bm25_retrieved_count", None),
                    "used_hyde": "hyde_document" in meta,
                    "route_reason": meta.get("route_reason", ""),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "request_error": request_error,
                    "answer": answer,
                    "retrieved_filenames_json": json.dumps(
                        retrieved_filenames, ensure_ascii=False
                    ),
                    "source_filenames_json": json.dumps(
                        source_filenames, ensure_ascii=False
                    ),
                    "sources_json": json.dumps(sources, ensure_ascii=False),
                    "meta_json": json.dumps(meta, ensure_ascii=False),
                }
            )

            time.sleep(CHAT_REQUEST_INTERVAL_SEC)

        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        self.write_single_summary(df, experiment_name)

        print(f"Saved experiment result: {output_csv_path}")
        return output_csv_path

    @staticmethod
    def write_single_summary(df: pd.DataFrame, experiment_name: str) -> Path:
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = SUMMARY_DIR / f"{experiment_name}.summary.csv"

        available_metrics = [
            metric for metric in SUMMARY_METRICS if metric in df.columns
        ]
        summary = (
            df.groupby(["experiment", "cluster", "question_type"], dropna=False)[
                available_metrics
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_path


class ExperimentSuite:
    def __init__(
        self, server_cmd: list[str] | None = None, use_running_server: bool = False
    ):
        self.config_manager = RagConfigManager()
        self.server_manager = ServerManager(
            server_cmd=server_cmd,
            use_running_server=use_running_server,
        )
        self.cluster_manager = ClusterManager()
        self.evaluator = EvaluationRunner()
        self.prepared_once = False
        self.use_running_server = use_running_server

    @staticmethod
    def validate_experiment_names(experiment_names: Iterable[str]) -> list[str]:
        names = list(experiment_names)
        unknown = [name for name in names if name not in EXPERIMENTS]

        if unknown:
            raise ValueError(
                f"Unknown experiments: {unknown}. Available: {list(EXPERIMENTS)}"
            )

        return names

    def run_one(
        self,
        experiment_name: str,
        reset_documents: bool = False,
        prepare_documents: bool = True,
    ) -> Path:
        self.config_manager.write_experiment_config(experiment_name)

        if self.use_running_server:
            print(
                "WARNING: --use-running-server is enabled. "
                "The running backend may not reload the just-written rag_config.json unless you restarted it manually."
            )

        self.server_manager.start(experiment_name)
        self.cluster_manager.tokens.clear()
        self.evaluator.tokens.clear()

        try:
            if prepare_documents and not self.prepared_once:
                self.cluster_manager.prepare_all_clusters(
                    reset_documents=reset_documents
                )
                self.prepared_once = True
            elif prepare_documents and self.prepared_once:
                print(
                    "Documents have already been prepared in this run. Skip document preparation."
                )

            return self.evaluator.run_experiment(experiment_name)
        finally:
            self.server_manager.stop()

    def run_many(
        self,
        experiment_names: list[str],
        reset_documents: bool = False,
        prepare_documents: bool = True,
        restore_config: bool = True,
    ) -> list[Path]:
        outputs: list[Path] = []

        try:
            for index, experiment_name in enumerate(experiment_names, start=1):
                print("\n" + "=" * 88)
                print(f"Running {index}/{len(experiment_names)}: {experiment_name}")
                print(EXPERIMENTS[experiment_name]["title"])
                print(EXPERIMENTS[experiment_name]["purpose"])
                print("=" * 88)

                outputs.append(
                    self.run_one(
                        experiment_name,
                        reset_documents=reset_documents,
                        prepare_documents=prepare_documents,
                    )
                )

            self.write_overall_summary(outputs)
            return outputs

        finally:
            self.server_manager.stop()
            if restore_config:
                self.config_manager.restore_original_config()

    @staticmethod
    def write_overall_summary(output_paths: list[Path]) -> None:
        if not output_paths:
            return

        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        frames = [pd.read_csv(path) for path in output_paths if path.exists()]
        if not frames:
            return

        df = pd.concat(frames, ignore_index=True)

        all_results_path = RESULT_DIR / "all_experiment_results.csv"
        df.to_csv(all_results_path, index=False, encoding="utf-8-sig")

        available_metrics = [
            metric for metric in SUMMARY_METRICS if metric in df.columns
        ]

        by_experiment = (
            df.groupby(["experiment", "experiment_title"], dropna=False)[
                available_metrics
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        experiment_summary_path = SUMMARY_DIR / "all_experiments.summary.csv"
        by_experiment.to_csv(experiment_summary_path, index=False, encoding="utf-8-sig")

        by_experiment_cluster = (
            df.groupby(["experiment", "cluster"], dropna=False)[available_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        cluster_summary_path = SUMMARY_DIR / "all_experiments.by_cluster.summary.csv"
        by_experiment_cluster.to_csv(
            cluster_summary_path, index=False, encoding="utf-8-sig"
        )

        by_experiment_type = (
            df.groupby(["experiment", "question_type"], dropna=False)[available_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        type_summary_path = SUMMARY_DIR / "all_experiments.by_question_type.summary.csv"
        by_experiment_type.to_csv(type_summary_path, index=False, encoding="utf-8-sig")

        print(f"Saved merged results: {all_results_path}")
        print(f"Saved overall summary: {experiment_summary_path}")
        print(f"Saved cluster summary: {cluster_summary_path}")
        print(f"Saved question type summary: {type_summary_path}")


def list_experiments() -> None:
    print("Configured experiments:\n")

    for name, experiment in EXPERIMENTS.items():
        rag = experiment["rag"]
        model_names = [role["model"] for role in rag["generation"]]

        print(f"{name}")
        print(f"  标题：{experiment['title']}")
        print(f"  论文位置：{experiment['paper_section']}")
        print(f"  目的：{experiment['purpose']}")
        print(f"  数据集：{experiment['dataset_path']}")
        print(f"  输出：{experiment['output_csv_path']}")
        print(
            f"  rag_config：hyde.enabled={rag['hyde_enabled']}，"
            f"routing.enabled={rag['routing_enabled']}，top_k={rag['top_k']}，"
            f"generation={model_names}"
        )
        print()


def make_dataset_templates(overwrite: bool = False) -> None:
    eval_dir = SCRIPT_DIR / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    main_template = []
    rejection_template = []

    for cluster_name, cluster in CLUSTERS.items():
        data_dir = Path(cluster["data_dir"])
        allowed = {ext.lower() for ext in cluster.get("allowed_extensions", [])}

        sample_files: list[str] = []
        if data_dir.exists():
            sample_files = sorted(
                [
                    path.name
                    for path in data_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in allowed
                ]
            )[:3]

        if not sample_files:
            fallback_ext = sorted(allowed)[0] if allowed else ".md"
            sample_files = [f"请替换为真实文件名{fallback_ext}"]

        for index, filename in enumerate(sample_files, start=1):
            main_template.append(
                {
                    "id": f"{cluster_name}_main_{index:03d}",
                    "cluster": cluster_name,
                    "question_type": "factual_lookup",
                    "query": "请替换为根据该文档可以回答的中文技术问题。",
                    "expected_doc_filename": filename,
                    "is_out_of_bounds": False,
                    "reference_answer": "请替换为参考答案。",
                    "evidence_hint": "请替换为人工检查依据。",
                }
            )

        rejection_template.append(
            {
                "id": f"{cluster_name}_reject_001",
                "cluster": cluster_name,
                "question_type": "out_of_bounds",
                "query": "请替换为明显不属于该知识库范围的问题。",
                "expected_doc_filename": "",
                "is_out_of_bounds": True,
                "reference_answer": "该问题超出当前知识库范围，系统不应强行引用文档。",
                "evidence_hint": "",
            }
        )

    templates = {
        MAIN_EVAL_PATH: main_template,
        REJECTION_EVAL_PATH: rejection_template,
    }

    for path, content in templates.items():
        if path.exists() and not overwrite:
            print(f"Template already exists, skip: {path}")
            continue

        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)

        print(f"Wrote editable template: {path}")


def check_required_paths() -> None:
    print("Checking required paths...\n")

    for cluster_name, cluster in CLUSTERS.items():
        data_dir = Path(cluster["data_dir"])
        allowed = {ext.lower() for ext in cluster.get("allowed_extensions", [])}

        files = []
        if data_dir.exists():
            files = [
                path
                for path in data_dir.iterdir()
                if path.is_file() and path.suffix.lower() in allowed
            ]

        status = "OK" if data_dir.exists() and files else "MISSING"
        print(
            f"{status} cluster={cluster_name} path={data_dir} "
            f"extensions={sorted(allowed)} file_count={len(files)}"
        )

    for path in [MAIN_EVAL_PATH, REJECTION_EVAL_PATH]:
        status = "OK" if path.exists() else "MISSING"
        count_text = ""

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                count_text = (
                    f" items={len(data) if isinstance(data, list) else 'not-list'}"
                )

                if isinstance(data, list):
                    EvaluationRunner.validate_dataset_items(data, path)
                    count_text += " schema=OK"

            except Exception as error:
                count_text = f" error={error}"

        print(f"{status} eval_dataset={path}{count_text}")

    print(f"rag_config_path={RAG_CONFIG_PATH} exists={RAG_CONFIG_PATH.exists()}")
    print("\nMetric note:")
    print(
        "  retrieval_document_hit requires backend meta['retrieved_filenames']. "
        "If this field is absent, retrieval metrics will be empty but citation metrics remain valid."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final RAG thesis experiments on Python 3.12 Chinese docs and MDN English docs."
    )

    parser.add_argument(
        "--all", action="store_true", help="Run all configured experiments."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        action="append",
        help="Run one experiment by name. Can be used multiple times.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all experiments and exit."
    )
    parser.add_argument(
        "--check", action="store_true", help="Check required paths and eval files."
    )
    parser.add_argument(
        "--make-templates",
        action="store_true",
        help="Create editable evaluation dataset templates.",
    )
    parser.add_argument(
        "--overwrite-templates",
        action="store_true",
        help="Overwrite existing dataset templates.",
    )
    parser.add_argument(
        "--reset-documents",
        action="store_true",
        help="Delete existing uploaded documents once before uploading local files again.",
    )
    parser.add_argument(
        "--skip-document-prepare",
        action="store_true",
        help="Do not register users or upload documents. Use only when documents are already prepared.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Start the server, register users, upload documents, wait for embeddings, then exit.",
    )
    parser.add_argument(
        "--keep-rag-config",
        action="store_true",
        help="Keep the last experiment's rag_config.json instead of restoring the original file.",
    )
    parser.add_argument(
        "--use-running-server",
        action="store_true",
        help="Use an already running FastAPI server instead of starting/stopping one.",
    )
    parser.add_argument(
        "--server-cmd",
        nargs=argparse.REMAINDER,
        help=(
            "Override the command used to start FastAPI. Example: "
            "--server-cmd python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_experiments()
        return

    if args.check:
        check_required_paths()
        return

    if args.make_templates:
        make_dataset_templates(overwrite=args.overwrite_templates)
        return

    server_cmd = args.server_cmd if args.server_cmd else None
    suite = ExperimentSuite(
        server_cmd=server_cmd,
        use_running_server=args.use_running_server,
    )

    if args.prepare_only:
        prep_experiment = "exp04_full_routing_top4"
        suite.config_manager.write_experiment_config(prep_experiment)
        suite.server_manager.start(prep_experiment)

        try:
            suite.cluster_manager.prepare_all_clusters(
                reset_documents=args.reset_documents
            )
        finally:
            suite.server_manager.stop()
            if not args.keep_rag_config:
                suite.config_manager.restore_original_config()

        return

    if args.all:
        experiment_names = list(EXPERIMENTS.keys())
    else:
        experiment_names = args.experiment or list(EXPERIMENTS.keys())

    experiment_names = ExperimentSuite.validate_experiment_names(experiment_names)

    suite.run_many(
        experiment_names,
        reset_documents=args.reset_documents,
        prepare_documents=not args.skip_document_prepare,
        restore_config=not args.keep_rag_config,
    )


if __name__ == "__main__":
    main()
