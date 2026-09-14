"""structured_cite.py — 可靠溯源标注
=====================================
让 LLM 在「生成答案」的同时，强制按结构输出三部分，并用 pydantic 做硬校验：

    CitedAnswer
      ├─ answer      : str            最终回答正文
      ├─ sources     : List[SourceRef]  实际引用的来源（编号 / 来源文档 / 原文片段 / 引用原因）
      └─ confidence  : float (0-1)    对「回答完全基于检索资料、无编造」的把握度

为什么可靠（三层保险）：
  1. response_format(json_schema) —— 尽量让 LLM 直接吐合规 JSON，减少解析失败；
  2. pydantic 模型校验            —— 无论 LLM 返回什么，过一遍模型，畸形/缺字段即降级；
  3. 最终降级                      —— 任何接口/解析/校验异常 → 退回普通回答 + 空来源 + 低置信，
                                     保证问答流程不中断、且来源永远可追溯（哪怕是"无可靠来源"）。

设计取舍：
  - openai 仅作类型提示（TYPE_CHECKING），运行时零额外依赖；真正调用靠传入的 client。
  - 优先 json_schema 严格模式；若该接口版本不支持，自动退回 json_object 重试。
"""
from __future__ import annotations

import json
from typing import List, Optional, TYPE_CHECKING

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - 兜底：openai 未带 pydantic 的极端情况
    BaseModel = object
    Field = lambda *a, **k: None
    _HAS_PYDANTIC = False

if TYPE_CHECKING:  # 避免运行时强依赖 openai
    from openai import OpenAI


# ========== 1) pydantic 模型：解析后硬校验 ==========
class SourceRef(BaseModel):
    ref_id: int = Field(..., description="对应上下文块的序号（1-based，与编号资料一致）")
    source: str = Field(..., description="来源文档名")
    excerpt: str = Field(..., description="被引用的原文片段（尽量照抄资料）")
    reason: Optional[str] = Field(default=None, description="为什么引用这段")


class CitedAnswer(BaseModel):
    answer: str = Field(..., description="最终回答正文")
    sources: List[SourceRef] = Field(default_factory=list, description="本回答实际引用的来源列表")
    confidence: float = Field(..., description="基于检索资料的把握程度 0-1（资料不足应给低分）")
    confidence_reason: Optional[str] = Field(default=None, description="置信度判定依据")


# ========== 2) JSON Schema：喂给 LLM 的 structured outputs 约束 ==========
CITED_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "integer"},
                    "source": {"type": "string"},
                    "excerpt": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["ref_id", "source", "excerpt"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
        "confidence_reason": {"type": "string"},
    },
    "required": ["answer", "sources", "confidence"],
    "additionalProperties": False,
}


# ========== 3) 构造「带编号资料」的 prompt ==========
def build_citation_prompt(question, contexts, max_chars: int = 800) -> str:
    blocks = []
    for i, c in enumerate(contexts):
        if isinstance(c, dict):
            src = c.get("source", "?")
            content = c.get("content", "")
        else:
            src = "?"
            content = str(c)
        blocks.append(f"[{i + 1}] (来源: {src})\n{content[:max_chars]}")
    ctx = "\n\n".join(blocks)
    return (
        "你是一个严谨的企业知识库问答助手。请 ONLY 基于下面带编号的资料回答用户问题。\n"
        "规则：\n"
        "1. 回答正文写入 answer 字段。\n"
        "2. 对回答中参考到的每条资料，在 sources 中给出：ref_id(资料编号)、source(来源文档名)、"
        "excerpt(被引用的原文片段，尽量照抄)、reason(引用原因)。\n"
        "3. 只能引用给出的编号资料，严禁编造编号外的来源。\n"
        "4. confidence 表示你对『回答完全基于上述资料、无编造』的把握(0-1)；资料不足请给低分，"
        "并在 confidence_reason 说明原因。\n\n"
        f"【资料】\n{ctx}\n\n"
        f"【用户问题】\n{question}"
    )


# ========== 4) 核心：生成带溯源结构的答案 ==========
# ========== 3.5) 任务型系统指令（prefix cache 友好：固定前缀，重复点击命中缓存） ==========
_DEFAULT_TASK_PROMPTS = {
    "revise_proposal": (
        "你是一名资深游戏策划评审专家。当用户提交一份策划案时，请严格依据知识库中检索到的"
        "『标准策划案模板 / 范例』审阅，并给出可执行的修改建议。重点关注："
        "1) 玩法描述是否足够细致——不只说明『有什么玩法』，更要拆解到『玩家每一步操作』："
        "输入方式、触发条件、反馈表现、状态流转是否都设计到位，有无遗漏的操作步骤或断开的交互闭环；"
        "2) 目标与定位是否清晰；"
        "3) 结构是否完整（背景、目标、玩法、数值、风险等模块是否齐全）；"
        "4) 设定是否自洽、有无逻辑漏洞或可行性风险；"
        "5) 表述是否专业、无歧义。"
        "输出先给总体评价，再按『问题 -> 建议 -> 修改示例』逐条列出；"
        "若用户尚未提交具体策划案内容，请先请其粘贴。"
    ),
    "revise_art": (
        "你是一名资深美术指导。当用户提交美术需求文档时，请依据知识库中的『标准美术需求模板』审阅，"
        "并给出可执行的修改建议。请逐部件（component）检查策划的描述是否细致："
        "1) 每个部件不只描述『大致外观』，还要明确其质感（材质 / 光影 / 笔触 / 精度层级）与细节精度要求；"
        "2) 该部件与游戏世界观、整体美术基调的搭配是否一致，有无风格冲突或调性割裂；"
        "3) 交付物清单（立绘 / 场景 / UI / 特效等）是否完整、规格（尺寸 / 格式 / 分辨率）是否清晰；"
        "4) 需求是否可被美术直接执行、无歧义；5) 工期与优先级是否合理。"
        "尤其留意策划常疏漏之处：世界观与视觉的冲突、质感层级缺失、部件间风格不统一。"
        "按『问题 -> 建议 -> 修改示例』输出；若用户未提交文档，请先请其粘贴。"
    ),
    "organize_ui": (
        "你是一名资深交互 / UI 设计专家。当用户提交 UI 需求时，请依据知识库中的『标准 UI 需求模板』帮助其梳理。"
        "请逐环节检查："
        "1) 不只罗列页面，更要明确每个界面元素的交互细节（操作方式、状态变化、反馈）；"
        "2) 信息架构与页面层级是否清晰；"
        "3) 核心交互流程是否闭环、有无遗漏状态（空 / 错 / 加载 / 成功）；"
        "4) 组件与规范是否统一；"
        "5) 是否遗漏埋点 / 无障碍 / 多端适配说明。"
        "输出『现状梳理 -> 缺失项 -> 整理后的需求框架』；若用户未提交内容，请先请其粘贴。"
    ),
}

# ========== 3.6) 任务提示词 / 标准文档：从 config/ 加载，支持管理员在线编辑 ==========
import os as _os

_CONFIG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config")

def _load_json_config(name, default):
    """加载 config/<name>.json；文件不存在或解析失败时回退 default。"""
    path = _os.path.join(_CONFIG_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return default

def _save_json_config(name, data):
    """写回 config/<name>.json（管理员编辑后持久化）。"""
    _os.makedirs(_CONFIG_DIR, exist_ok=True)
    path = _os.path.join(_CONFIG_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 运行时任务提示词：优先用 config/prompts.json，缺失时回退到上方默认字典
TASK_SYSTEM_PROMPTS = _load_json_config("prompts.json", _DEFAULT_TASK_PROMPTS)
# 标准文档（并入前缀缓存的参考基准），由管理员在 /admin 页面维护
STANDARD_DOCS = _load_json_config("standard_docs.json", {})

def get_standard_doc(task):
    """返回某任务对应的标准文档文本（用于并入前缀缓存）。"""
    return STANDARD_DOCS.get(task, "")

def get_task_prompts():
    """返回当前任务提示词字典的快照（与 get_standard_docs 同理，避免持有过期引用）。"""
    return dict(TASK_SYSTEM_PROMPTS)

def get_standard_docs():
    """返回当前标准文档字典的快照（与 get_task_prompt 同理，避免调用方持有过期引用）。

    调用方要做「读旧值 → 合并 → 写回」时，必须用本函数取旧值：
    直接 import STANDARD_DOCS 拿到的是导入那一刻的字典对象，
    save_standard_docs() 之后它就过期了，再写回会把别人刚保存的内容覆盖掉。
    """
    return dict(STANDARD_DOCS)

def get_task_prompt(task):
    """返回某任务对应的系统指令；任务不存在时返回 None。

    ⚠️ 为什么必须用函数取值，而不是 `from structured_cite import TASK_SYSTEM_PROMPTS`？
    save_prompts() 内部用 `global TASK_SYSTEM_PROMPTS` 重新绑定了一个新字典，
    而 `from ... import` 只在导入时取值一次，外部模块持有的仍是旧字典对象，
    会导致管理员在 /admin/prompts 保存后「页面提示已保存」但线上仍用旧提示词。
    通过函数访问模块级变量，才能每次都读到最新的绑定。
    """
    if not task:
        return None
    return TASK_SYSTEM_PROMPTS.get(task)

def save_prompts(data):
    """管理员更新任务提示词并持久化到 config/prompts.json。"""
    global TASK_SYSTEM_PROMPTS
    TASK_SYSTEM_PROMPTS = dict(data)
    _save_json_config("prompts.json", TASK_SYSTEM_PROMPTS)

def save_standard_docs(data):
    """管理员更新标准文档并持久化到 config/standard_docs.json。"""
    global STANDARD_DOCS
    STANDARD_DOCS = dict(data)
    _save_json_config("standard_docs.json", STANDARD_DOCS)

# DeepSeek 前缀缓存开关：把固定的 system 指令作为请求开头，重复请求可命中缓存省 token。
# 切到非 DeepSeek 模型时把此值设为 None 即可关闭（避免 unknown 参数报错）。
PREFIX_CACHE_P = 1.0

def _cache_kwargs(model=None):
    """返回启用 DeepSeek 前缀缓存所需的额外请求参数；非 DeepSeek 模型返回空字典。"""
    if PREFIX_CACHE_P is not None and model and "deepseek" in str(model).lower():
        return {"extra_body": {"cache_p": PREFIX_CACHE_P}}
    return {}


def generate_cited_answer(client, model, question, contexts, max_chars: int = 800,
                          system_instruction: str = None) -> "CitedAnswer":
    """调用 LLM 生成结构化溯源答案。

    返回 CitedAnswer（pydantic 校验过的实例）。任何失败都降级为：
       answer=普通文本 / sources=[] / confidence=0.2 / confidence_reason=失败说明
    """
    prompt = build_citation_prompt(question, contexts, max_chars=max_chars)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    cache_kw = _cache_kwargs(model)

    # 优先 json_schema 严格模式；若该接口不支持则退回 json_object
    attempts = [
        {"type": "json_schema", "json_schema": {"name": "CitedAnswer", "strict": True, "schema": CITED_ANSWER_SCHEMA}},
        {"type": "json_object"},
    ]
    last_err: Exception | None = None
    for fmt in attempts:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                timeout=20,
                response_format=fmt,
                **cache_kw,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            return CitedAnswer.model_validate(data)
        except Exception as e:  # 接口报错 / JSON 解析失败 / pydantic 校验失败
            last_err = e
            continue

    # 结构化全部失败 → 退回普通生成，保证有回答
    try:
        fb_messages = []
        if system_instruction:
            fb_messages.append({"role": "system", "content": system_instruction})
        fb_messages.append({"role": "user", "content": f"请基于以下资料回答用户问题：\n{prompt}"})
        resp = client.chat.completions.create(
            model=model,
            messages=fb_messages,
            temperature=0.3,
            timeout=20,
            **cache_kw,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        text = "抱歉，生成回答时出现问题，请稍后重试。"
    return CitedAnswer(
        answer=text,
        sources=[],
        confidence=0.2,
        confidence_reason=f"结构化生成失败（{type(last_err).__name__}），已降级为普通回答",
    )
