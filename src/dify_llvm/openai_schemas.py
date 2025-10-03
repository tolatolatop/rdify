import time
from uuid import uuid4
from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field

def generate_id():
    return str(uuid4())

def create_time():
    return int(time.time())


class ModelCapabilities(BaseModel):
    chat: bool = Field(..., title="是否支持 chat", description="是否支持 chat", default_factory=lambda: False)
    completion: bool = Field(..., title="是否支持 completion", description="是否支持 completion", default_factory=lambda: False)
    stream: bool = Field(..., title="是否支持 stream", description="是否支持 stream", default_factory=lambda: False)

class ModelInfo(BaseModel):
    id: str = Field(..., title="模型 ID", description="模型的唯一标识符", default_factory=generate_id)
    object: Literal["model"] = Field("model", description="对象类型，总为 \"model\"")
    owned_by: Optional[str] = Field(None, title="拥有者", description="该模型所属的账户或组织")
    capabilities: ModelCapabilities = Field(
        default_factory=ModelCapabilities,
        title="能力描述",
        description="该模型支持的能力（如 chat、completion、streaming、最大 token 数等）"
    )


class Usage(BaseModel):
    prompt_tokens: Optional[int] = Field(None, title="Prompt Tokens 数量", description="提示（输入）使用的 token 数")
    completion_tokens: Optional[int] = Field(None, title="Completion Tokens 数量", description="模型生成（输出）使用的 token 数")
    total_tokens: Optional[int] = Field(None, title="总 Tokens 数量", description="prompt + completion 的总 token 数")


class ChoiceDeltaContent(BaseModel):
    content: Optional[str] = Field(None, title="增量内容", description="流式输出时本次 chunk 的内容")
    role: Optional[Literal["system", "user", "assistant", "function"]] = Field(
        None, title="角色", description="本次 chunk 若包含角色切换，则指明角色"
    )
    # 如果你支持函数调用（function_call），可以继续加字段 name, arguments 等
    # name: Optional[str] = Field(None, title="函数名", description="在 function-calling 模式下的函数名")
    # arguments: Optional[str] = Field(None, title="函数调用参数（JSON 字符串）", description="函数调用的参数内容")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., title="角色", description="消息的角色（system / user / assistant / function）"
    )
    content: Optional[str] = Field(
        None, title="内容", description="消息文本内容"
    )
    name: Optional[str] = Field(
        None, title="名称", description="仅在 function-calling 场景下使用，代表消息所属的函数名"
    )


class ChatCompletionChoice(BaseModel):
    index: int = Field(..., title="Choice 索引", description="该选项在返回数组中的索引（0 开始）")
    message: ChatMessage = Field(..., title="消息", description="完整生成的消息（非流式模式下）")
    finish_reason: Optional[str] = Field(
        None, title="结束原因", description="为何结束生成（如 'stop'、'length'、'function_call' 等）"
    )
    delta: Optional[ChoiceDeltaContent] = Field(
        title="增量内容", description="流式模式下的本次 chunk 内容差异 (delta)", default_factory=lambda: ChoiceDeltaContent(content="", role="")
    )


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., title="响应 ID", description="本次 API 调用的唯一标识符", default_factory=generate_id)
    object: Literal["chat.completion"] = Field(
        "chat.completion",
        title="对象类型",
        description="固定为 \"chat.completion\""
    )
    created: int = Field(..., title="创建时间 (Unix)", description="响应生成时间戳（Unix 秒级）", default_factory=create_time)
    model: str = Field(..., title="模型 ID", description="用于推理的模型标识符")
    usage: Optional[Usage] = Field(None, title="使用情况统计", description="prompt / completion / total token 用量")
    choices: List[ChatCompletionChoice] = Field(
        ..., title="候选项列表", description="生成的多个候选消息选项", default_factory=list
    )


class CompletionChoice(BaseModel):
    index: int = Field(..., title="Choice 索引", description="该选项在返回数组中的索引（0 开始）")
    text: str = Field(..., title="生成文本", description="模型生成的文本内容")
    logprobs: Optional[Any] = Field(
        None, title="Log 概率", description="若请求时带 logprobs 参数，则返回 token 级别的 log 概率信息"
    )
    finish_reason: Optional[str] = Field(
        None, title="结束原因", description="为何结束生成（如 'stop'、'length' 等）"
    )


class CompletionResponse(BaseModel):
    id: str = Field(..., title="响应 ID", description="本次 API 调用的唯一标识符", default_factory=generate_id)
    object: Literal["text_completion"] = Field(
        "text_completion",
        title="对象类型",
        description="固定为 \"text_completion\""
    )
    created: int = Field(..., title="创建时间 (Unix)", description="响应生成时间戳（Unix 秒级）", default_factory=create_time)
    model: str = Field(..., title="模型 ID", description="用于推理的模型标识符")
    usage: Optional[Usage] = Field(None, title="使用情况统计", description="prompt / completion / total token 用量")
    choices: List[CompletionChoice] = Field(
        ..., title="候选项列表", description="生成的多个候选文本选项"
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., title="模型 ID", description="指定要调用的模型标识符")
    messages: List[ChatMessage] = Field(..., title="消息列表", description="对话历史消息列表，从 system/user 开始")
    temperature: Optional[float] = Field(
        None, title="温度", description="控制采样随机性，越大越随机"
    )
    top_p: Optional[float] = Field(
        None, title="Top-p (nucleus) 采样参数", description="控制累积概率上限的采样方法"
    )
    n: Optional[int] = Field(
        None, title="返回候选数量", description="一次性希望返回多少条候选结果"
    )
    stream: Optional[bool] = Field(
        False, title="是否流式", description="是否以流 (chunked) 方式返回响应"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, title="终止符", description="生成时遇到这些 token 或字符串即停止"
    )
    max_tokens: Optional[int] = Field(
        None, title="最大生成长度", description="最多生成多少个 token"
    )
    presence_penalty: Optional[float] = Field(
        None, title="存在惩罚项", description="控制生成中重复内容的惩罚强度"
    )
    frequency_penalty: Optional[float] = Field(
        None, title="频率惩罚项", description="根据 token 出现频率惩罚重复"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, title="Logit 偏好", description="对特定 token 的 logit 值加偏置"
    )
    user: Optional[str] = Field(
        None, title="用户标识", description="调用方提供的用户 ID（用于审计 / 日志）"
    )
    # 若支持 function-calling，可加以下字段：
    # functions: Optional[List[FunctionSpec]] = Field(None, title="函数定义列表", description="可调用的函数接口定义")
    # function_call: Optional[Union[Literal["none","auto"], Dict[str,Any]]] = Field(None, title="函数调用控制", description="控制函数调用行为 (none / auto / 指定函数名)")


class CompletionRequest(BaseModel):
    model: str = Field(..., title="模型 ID", description="指定要调用的模型标识符")
    prompt: Union[str, List[str]] = Field(
        ..., title="提示 (prompt)", description="输入提示，可以是单个字符串或多个字符串"
    )
    suffix: Optional[str] = Field(
        None, title="后缀 (suffix)", description="在生成后追加的字符串"
    )
    max_tokens: Optional[int] = Field(
        None, title="最大生成长度", description="最多生成多少个 token"
    )
    temperature: Optional[float] = Field(
        None, title="温度", description="控制采样随机性，越大越随机"
    )
    top_p: Optional[float] = Field(
        None, title="Top-p (nucleus) 采样参数", description="控制累积概率上限的采样方法"
    )
    n: Optional[int] = Field(
        None, title="返回候选数量", description="一次性希望返回多少条候选结果"
    )
    stream: Optional[bool] = Field(
        False, title="是否流式", description="是否以流 (chunked) 方式返回响应"
    )
    logprobs: Optional[int] = Field(
        None, title="返回 log 概率层级", description="若设定非 None，则返回 token 级别的 log 概率"
    )
    echo: Optional[bool] = Field(
        False, title="是否回显 prompt", description="是否将 prompt 内容也包含在返回文本中"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, title="终止符", description="生成时遇到这些 token 或字符串即停止"
    )
    presence_penalty: Optional[float] = Field(
        None, title="存在惩罚项", description="控制生成中重复内容的惩罚强度"
    )
    frequency_penalty: Optional[float] = Field(
        None, title="频率惩罚项", description="根据 token 出现频率惩罚重复"
    )
    best_of: Optional[int] = Field(
        None, title="最佳采样数", description="从多个生成候选中选择最优（只在 非流式 且 n=None 时有效）"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, title="Logit 偏好", description="对特定 token 的 logit 值加偏置"
    )
    user: Optional[str] = Field(
        None, title="用户标识", description="调用方提供的用户 ID（用于审计 / 日志）"
    )


class ListModelsResponse(BaseModel):
    data: List[ModelInfo] = Field(..., title="模型列表", description="支持的模型列表")

class GetModelResponse(ModelInfo):
    pass
