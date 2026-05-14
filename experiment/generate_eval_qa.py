#!/usr/bin/env python3
"""
Generate evaluation Q&A datasets for the RAG thesis experiments.

This version is adapted for the two new document collections:

1. python_zh_312_html
   - Python 3.12 Chinese official documentation
   - Source files: .html

2. mdn_web_docs_en_md
   - MDN Web Docs English documentation
   - Source files: .md

Key changes from the old script:
- Removed answer_keywords from generated samples.
- Supports both HTML and Markdown source files.
- Keeps useful code content when cleaning Markdown instead of deleting fenced code blocks.
- Uses cluster-specific out-of-scope questions to avoid accidental in-scope rejection cases.
- Adds evidence_hint for human inspection only. It is not intended for automatic scoring.

Recommended location:
    experiment/generate_eval_qa.py

Run from project root:
    python experiment/generate_eval_qa.py --main-per-cluster 30 --rejection-per-cluster 10 --overwrite

Outputs:
    experiment/data/eval/main_eval.json
    experiment/data/eval/rejection_eval.json
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAG_CONFIG_PATH = PROJECT_ROOT / "rag_config.json"
EVAL_DIR = SCRIPT_DIR / "data" / "eval"
MAIN_EVAL_PATH = EVAL_DIR / "main_eval.json"
REJECTION_EVAL_PATH = EVAL_DIR / "rejection_eval.json"


CLUSTERS: dict[str, dict[str, Any]] = {
    "python_zh_312_html": {
        "display_name": "Python 3.12 中文官方技术文档",
        "data_dir": SCRIPT_DIR / "data" / "clusters" / "python_zh_312_html",
        "allowed_extensions": [".html"],
        "domain_hint": (
            "Python 语言基础、标准库、语言参考、安装使用、模块管理、"
            "解释器用法、异常处理、面向对象、并发与常用编程任务"
        ),
        "source_language": "中文",
        "question_language": "中文",
    },
    "mdn_web_docs_en_md": {
        "display_name": "MDN Web Docs 英文技术文档",
        "data_dir": SCRIPT_DIR / "data" / "clusters" / "mdn_web_docs_en_md",
        "allowed_extensions": [".md"],
        "domain_hint": (
            "Web 前端技术、HTML、CSS、JavaScript、HTTP、Web API、"
            "浏览器行为、页面结构、样式规则与脚本编程"
        ),
        "source_language": "英文",
        "question_language": "中文",
    },
}


QUESTION_TYPE_PLAN: list[tuple[str, int]] = [
    ("factual_lookup", 10),
    ("procedure_usage", 8),
    ("configuration_troubleshooting", 6),
    ("conceptual_summary", 4),
    ("comparison_reasoning", 2),
]


TYPE_HINTS: dict[str, str] = {
    "factual_lookup": "生成一个事实查询类问题，答案应能从文档中的定义、参数、规则、对象、函数、属性或功能说明中直接得到。",
    "procedure_usage": "生成一个操作或用法类问题，答案应涉及步骤、调用方式、配置方法、使用流程或代码组织方式。",
    "configuration_troubleshooting": "生成一个配置或排障类问题，答案应涉及错误原因、限制条件、检查项、兼容性或处理方法。",
    "conceptual_summary": "生成一个概念总结类问题，答案应要求概括某个机制、组件、功能或设计原则的作用。",
    "comparison_reasoning": "生成一个轻量比较或辨析类问题，答案应基于文档内容比较两个概念、选项、行为或使用场景。",
}


# Rejection questions are cluster-specific. A question can be valid technical content in another
# dataset, but it should be out of scope for the current cluster.
REJECTION_QUESTIONS: dict[str, list[str]] = {
    "python_zh_312_html": [
        "如何使用 CSS Grid 创建两列布局并设置列间距？",
        "HTTP 的 Cache-Control 头部中 max-age 和 no-cache 有什么区别？",
        "JavaScript 中 Promise.all 和 Promise.race 的适用场景有什么不同？",
        "HTML 表单中 label 元素和 input 元素应如何关联？",
        "浏览器中的 Same-Origin Policy 主要限制了哪些跨源访问？",
        "如何使用 Web API 的 Fetch 接口发送 POST 请求？",
        "CSS 中 position: sticky 的生效条件是什么？",
        "请说明 Flexbox 中 justify-content 和 align-items 的区别。",
        "如何在 Service Worker 中拦截网络请求并返回缓存资源？",
        "WebSocket 连接建立后客户端和服务端如何进行双向通信？",
        "如何在 PostgreSQL 中创建索引并查看查询计划？",
        "如何使用 Blender 设置玻璃材质的折射效果？",
        "请说明《红楼梦》中贾宝玉人物形象的文学意义。",
        "如何用 Excel 数据透视表统计不同月份的销售额？",
        "美国个人所得税申报中 W-2 表格和 1099 表格有什么区别？",
    ],
    "mdn_web_docs_en_md": [
        "Python 中 pathlib.Path.glob 和 rglob 有什么区别？",
        "Python 3.12 中如何创建虚拟环境并安装第三方包？",
        "Python 标准库 argparse 如何定义可选参数和位置参数？",
        "Python 中类方法、静态方法和实例方法有什么区别？",
        "Python 的 try、except、else 和 finally 分别适用于什么场景？",
        "如何使用 Python 的 asyncio 创建并发任务？",
        "Python 中列表推导式和生成器表达式有什么区别？",
        "Python 标准库 json 如何序列化自定义对象？",
        "Python 的 dataclass 适合解决什么类型的数据建模问题？",
        "Python 模块导入时 __name__ == '__main__' 有什么作用？",
        "如何在 MySQL 中设计电商订单表和库存表？",
        "请说明《红楼梦》中贾宝玉人物形象的文学意义。",
        "如何使用 Excel 数据透视表统计不同月份的销售额？",
        "美国个人所得税申报中 W-2 表格和 1099 表格有什么区别？",
        "如何在 Blender 中制作玻璃材质并设置环境光照？",
    ],
}


def load_dotenv(dotenv_path: Path) -> dict[str, str]:
    env_values: dict[str, str] = {}
    if not dotenv_path.exists():
        return env_values

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_values[key.strip()] = value.strip().strip('"').strip("'")

    return env_values


def get_env_value(key: str, dotenv_values: dict[str, str]) -> str | None:
    candidates = [key, key.upper(), key.lower()]
    for candidate in candidates:
        if os.environ.get(candidate):
            return os.environ[candidate]
        if dotenv_values.get(candidate):
            return dotenv_values[candidate]
    return None


def load_llm_config(
    provider_name: str = "doubao",
    model_name: str = "deepseek-v3.2",
) -> dict[str, str] | None:
    if not RAG_CONFIG_PATH.exists():
        print(
            f"rag_config.json not found: {RAG_CONFIG_PATH}. LLM generation will be skipped."
        )
        return None

    with open(RAG_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    provider = config.get("providers", {}).get(provider_name)
    if not provider:
        print(
            f"Provider not found in rag_config.json: {provider_name}. LLM generation will be skipped."
        )
        return None

    dotenv_values = load_dotenv(PROJECT_ROOT / ".env")
    env_key = f"{provider_name.replace('-', '_')}_api_key"
    api_key = provider.get("api_key") or get_env_value(env_key, dotenv_values)
    if not api_key:
        print(
            f"API key not found for {provider_name}. Expected {env_key} in .env or environment. LLM generation will be skipped."
        )
        return None

    return {
        "base_url": provider["base_url"].rstrip("/"),
        "api_key": api_key,
        "model": model_name,
    }


def strip_front_matter(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5 :]
    return text


def clean_markdown(text: str) -> str:
    text = strip_front_matter(text)

    # Keep code content because technical docs often carry crucial evidence in examples.
    text = re.sub(r"```[A-Za-z0-9_-]*\n(.*?)```", r"\1", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#>*_{}|]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_html(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)

    # Prefer the main body area when the docs expose it. This removes much of the sidebar noise.
    body_candidates = [
        r'(?is)<div[^>]+role=["\']main["\'][^>]*>(.*?)</div>\s*</section>',
        r"(?is)<main[^>]*>(.*?)</main>",
        r"(?is)<article[^>]*>(.*?)</article>",
        r'(?is)<div[^>]+class=["\'][^"\']*body[^"\']*["\'][^>]*>(.*?)</div>',
    ]
    for pattern in body_candidates:
        match = re.search(pattern, text)
        if match:
            text = match.group(1)
            break

    # Put line breaks around common block elements before stripping tags.
    text = re.sub(r"(?i)</(p|div|section|article|h1|h2|h3|h4|li|pre|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_markdown_title(raw: str, fallback: str) -> str:
    front_match = re.match(r"(?s)^---\s*(.*?)\s*---", raw)
    if front_match:
        for line in front_match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "title":
                title = value.strip().strip('"').strip("'")
                if title:
                    return title[:100]

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title[:100]

    return Path(fallback).stem[:100]


def extract_html_title(raw: str, fallback: str) -> str:
    for pattern in [
        r"(?is)<h1[^>]*>(.*?)</h1>",
        r"(?is)<title[^>]*>(.*?)</title>",
    ]:
        match = re.search(pattern, raw)
        if match:
            title = clean_html(match.group(1))
            title = re.sub(r"\s+—\s+Python.*$", "", title)
            title = re.sub(r"\s+-\s+Python.*$", "", title)
            title = title.strip()
            if title:
                return title[:100]

    return Path(fallback).stem[:100]


def read_doc_excerpt(path: Path, max_chars: int = 4200) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()

    if suffix == ".html":
        title = extract_html_title(raw, path.name)
        cleaned = clean_html(raw)
    elif suffix in {".md", ".mdx"}:
        title = extract_markdown_title(raw, path.name)
        cleaned = clean_markdown(raw)
    else:
        title = Path(path.name).stem
        cleaned = re.sub(r"\s+", " ", raw).strip()

    # Avoid sending extremely short or empty excerpts to the LLM.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return title, cleaned[:max_chars]


def list_cluster_files(cluster_name: str) -> list[Path]:
    cluster = CLUSTERS[cluster_name]
    data_dir = cluster["data_dir"]
    allowed_extensions = {ext.lower() for ext in cluster["allowed_extensions"]}

    if not data_dir.exists():
        raise FileNotFoundError(f"Cluster data directory not found: {data_dir}")

    files = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ],
        key=lambda path: path.name,
    )

    if not files:
        raise FileNotFoundError(
            f"No files with extensions {sorted(allowed_extensions)} found in {data_dir}"
        )

    return files


def build_type_sequence(main_per_cluster: int) -> list[str]:
    sequence: list[str] = []
    base_total = sum(count for _, count in QUESTION_TYPE_PLAN)

    for question_type, count in QUESTION_TYPE_PLAN:
        adjusted = max(1, round(count / base_total * main_per_cluster))
        sequence.extend([question_type] * adjusted)

    if len(sequence) < main_per_cluster:
        cycle = [question_type for question_type, _ in QUESTION_TYPE_PLAN]
        while len(sequence) < main_per_cluster:
            sequence.append(cycle[len(sequence) % len(cycle)])

    return sequence[:main_per_cluster]


def call_openai_compatible_json(
    llm_config: dict[str, str],
    messages: list[dict[str, str]],
    timeout: int = 90,
) -> dict[str, Any]:
    url = f"{llm_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_config["model"],
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Some OpenAI-compatible providers may still wrap JSON in a code fence.
        cleaned = content.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


def generate_item_with_llm(
    llm_config: dict[str, str],
    cluster_name: str,
    filename: str,
    title: str,
    excerpt: str,
    question_type: str,
) -> dict[str, Any]:
    cluster = CLUSTERS[cluster_name]

    system_prompt = (
        "你是技术文档 RAG 系统的评测集构建助手。"
        "你需要根据给定文档片段生成一个中文问答样本。"
        "问题必须能从文档片段中直接得到答案，不能依赖外部知识。"
        "问题应模拟真实用户提问，避免直接照抄标题或文件名。"
        "参考答案必须忠于文档片段，不要加入文档外推断。"
        "evidence_hint 只用于人工检查，应简短说明答案依据来自文档中的哪类信息，不要长篇引用原文。"
        "不要输出 answer_keywords 字段。"
        "只输出 JSON 对象，不要输出解释。"
    )

    user_prompt = f"""
文档集：{cluster['display_name']}
文档语言：{cluster['source_language']}
问题语言：{cluster['question_language']}
领域提示：{cluster['domain_hint']}
文件名：{filename}
文档标题：{title}
问题类型：{question_type}
问题类型要求：{TYPE_HINTS[question_type]}

文档片段：
{excerpt}

请输出如下 JSON 结构：
{{
  "query": "一个中文技术问题",
  "reference_answer": "基于文档片段给出的中文参考答案，控制在80到180字之间",
  "evidence_hint": "一句话说明该答案依据文档片段中的哪些信息",
  "question_type": "{question_type}"
}}
""".strip()

    data = call_openai_compatible_json(
        llm_config,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    query = str(data.get("query", "")).strip()
    answer = str(data.get("reference_answer", "")).strip()
    evidence_hint = str(data.get("evidence_hint", "")).strip()
    output_question_type = str(data.get("question_type") or question_type).strip()

    if not query or not answer:
        raise ValueError(f"Invalid LLM output: {data}")

    return {
        "cluster": cluster_name,
        "question_type": output_question_type,
        "query": query,
        "expected_doc_filename": filename,
        "is_out_of_bounds": False,
        "reference_answer": answer,
        "evidence_hint": evidence_hint,
    }


def first_sentence(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    candidate = parts[0] if parts else text
    if len(candidate) < 40 and len(parts) > 1:
        candidate = candidate + " " + parts[1]
    return candidate[:max_chars]


def generate_item_heuristic(
    cluster_name: str,
    filename: str,
    title: str,
    excerpt: str,
    question_type: str,
) -> dict[str, Any]:
    reference = first_sentence(excerpt, max_chars=180) or title
    title_for_query = title or Path(filename).stem

    query_templates = {
        "factual_lookup": f"{title_for_query} 中提到的核心概念或功能是什么？",
        "procedure_usage": f"在 {title_for_query} 相关场景中，用户应如何完成对应操作？",
        "configuration_troubleshooting": f"根据 {title_for_query}，出现相关配置或运行问题时应检查哪些内容？",
        "conceptual_summary": f"请概括 {title_for_query} 所说明机制的主要作用。",
        "comparison_reasoning": f"根据 {title_for_query}，相关概念或用法之间有什么区别？",
    }

    return {
        "cluster": cluster_name,
        "question_type": question_type,
        "query": query_templates.get(
            question_type, f"请说明 {title_for_query} 的主要内容。"
        ),
        "expected_doc_filename": filename,
        "is_out_of_bounds": False,
        "reference_answer": reference,
        "evidence_hint": f"依据文档《{title_for_query}》的正文片段生成。",
        "generation_note": "heuristic_fallback",
    }


def sample_files(files: list[Path], count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if count <= len(files):
        return rng.sample(files, count)

    repeated: list[Path] = []
    while len(repeated) < count:
        shuffled = files[:]
        rng.shuffle(shuffled)
        repeated.extend(shuffled)

    return repeated[:count]


def generate_main_eval(
    main_per_cluster: int,
    use_llm: bool,
    seed: int,
    sleep_sec: float,
    overwrite: bool,
    provider: str,
    model: str,
) -> list[dict[str, Any]]:
    if MAIN_EVAL_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"main_eval.json already exists: {MAIN_EVAL_PATH}. Use --overwrite to replace it."
        )

    llm_config = load_llm_config(provider, model) if use_llm else None
    if use_llm and llm_config is None:
        print("Falling back to heuristic generation because LLM config is unavailable.")

    items: list[dict[str, Any]] = []

    for cluster_index, cluster_name in enumerate(CLUSTERS):
        files = list_cluster_files(cluster_name)
        selected_files = sample_files(files, main_per_cluster, seed + cluster_index)
        type_sequence = build_type_sequence(main_per_cluster)

        for index, (file_path, question_type) in enumerate(
            tqdm(
                list(zip(selected_files, type_sequence)),
                desc=f"main_eval:{cluster_name}",
            ),
            start=1,
        ):
            title, excerpt = read_doc_excerpt(file_path)
            item = None

            if llm_config is not None:
                try:
                    item = generate_item_with_llm(
                        llm_config=llm_config,
                        cluster_name=cluster_name,
                        filename=file_path.name,
                        title=title,
                        excerpt=excerpt,
                        question_type=question_type,
                    )
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)
                except Exception as error:
                    print(
                        f"LLM generation failed for {file_path.name}: {error}. Use heuristic fallback."
                    )

            if item is None:
                item = generate_item_heuristic(
                    cluster_name=cluster_name,
                    filename=file_path.name,
                    title=title,
                    excerpt=excerpt,
                    question_type=question_type,
                )

            item["id"] = f"{cluster_name}_main_{index:03d}"
            items.append(item)

    return items


def generate_rejection_eval(
    rejection_per_cluster: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if REJECTION_EVAL_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"rejection_eval.json already exists: {REJECTION_EVAL_PATH}. Use --overwrite to replace it."
        )

    items: list[dict[str, Any]] = []

    for cluster_name in CLUSTERS:
        questions = REJECTION_QUESTIONS.get(cluster_name, [])
        if not questions:
            raise ValueError(
                f"No rejection questions configured for cluster: {cluster_name}"
            )

        for index in range(rejection_per_cluster):
            query = questions[index % len(questions)]
            items.append(
                {
                    "id": f"{cluster_name}_reject_{index + 1:03d}",
                    "cluster": cluster_name,
                    "question_type": "out_of_bounds",
                    "query": query,
                    "expected_doc_filename": "",
                    "is_out_of_bounds": True,
                    "reference_answer": "该问题超出当前知识库范围，系统不应强行引用文档。",
                    "evidence_hint": "",
                }
            )

    return items


def validate_items(items: list[dict[str, Any]]) -> None:
    required_keys = {
        "id",
        "cluster",
        "question_type",
        "query",
        "expected_doc_filename",
        "is_out_of_bounds",
        "reference_answer",
    }

    ids: set[str] = set()
    for item in items:
        missing = required_keys - set(item)
        if missing:
            raise ValueError(f"Item missing keys {missing}: {item}")

        if "answer_keywords" in item:
            raise ValueError(f"answer_keywords should not appear in item: {item['id']}")

        if item["id"] in ids:
            raise ValueError(f"Duplicate item id: {item['id']}")
        ids.add(item["id"])

        if item["cluster"] not in CLUSTERS:
            raise ValueError(f"Unknown cluster in item {item['id']}: {item['cluster']}")

        if not str(item["query"]).strip():
            raise ValueError(f"Empty query in item: {item['id']}")

        if (
            not item["is_out_of_bounds"]
            and not str(item["expected_doc_filename"]).strip()
        ):
            raise ValueError(
                f"Normal item must have expected_doc_filename: {item['id']}"
            )


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_items(data)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} items: {path}")


def print_summary(
    main_items: list[dict[str, Any]],
    rejection_items: list[dict[str, Any]],
) -> None:
    print("\nDataset summary:")

    for name, items in [("main_eval", main_items), ("rejection_eval", rejection_items)]:
        by_cluster: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for item in items:
            by_cluster[item["cluster"]] = by_cluster.get(item["cluster"], 0) + 1
            by_type[item["question_type"]] = by_type.get(item["question_type"], 0) + 1

        print(f"{name}: total={len(items)}, by_cluster={by_cluster}, by_type={by_type}")

    print("\nSample main item:")
    if main_items:
        print(json.dumps(main_items[0], ensure_ascii=False, indent=2))

    print("\nSample rejection item:")
    if rejection_items:
        print(json.dumps(rejection_items[0], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q&A evaluation datasets for the new Python + MDN RAG experiments."
    )
    parser.add_argument(
        "--main-per-cluster",
        type=int,
        default=30,
        help="Normal Q&A count per cluster. Recommended: 30 to 50.",
    )
    parser.add_argument(
        "--rejection-per-cluster",
        type=int,
        default=10,
        help="Out-of-scope question count per cluster. Recommended: 10.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM generation and use heuristic fallback only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing main_eval.json and rejection_eval.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document sampling.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between LLM calls.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="doubao",
        help="Provider name in rag_config.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3.2",
        help="Model name under the selected provider in rag_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_items = generate_main_eval(
        main_per_cluster=args.main_per_cluster,
        use_llm=not args.no_llm,
        seed=args.seed,
        sleep_sec=args.sleep,
        overwrite=args.overwrite,
        provider=args.provider,
        model=args.model,
    )

    rejection_items = generate_rejection_eval(
        rejection_per_cluster=args.rejection_per_cluster,
        overwrite=args.overwrite,
    )

    write_json(MAIN_EVAL_PATH, main_items)
    write_json(REJECTION_EVAL_PATH, rejection_items)
    print_summary(main_items, rejection_items)


if __name__ == "__main__":
    main()
