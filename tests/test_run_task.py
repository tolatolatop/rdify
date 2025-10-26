import pickle
from pathlib import Path
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from rdify.config import logs_dir
from rdify.apps.run_task_llm import check_run_task_is_finished
from rdify.apps import run_task_llm

def test_check_run_task_is_finished():
    task_log = """
【日志开始】
时间戳: 2024-05-21 09:00:00
分段: 目标
内容:
目标名称: 完成线上订单系统“支付模块”的高压性能验收
成功标准:
1. 在2000并发、支付峰值1200 TPS的条件下，99%订单支付耗时≤500 ms；
2. 无资金差错、无卡单、核心链路 P99 延迟下降30%。

---

【日志继续】
时间戳: 2024-05-21 09:05:00
分段: 计划`
内容:
计划编号: P-20240521-A
任务总数: 5
任务清单:
1. 【T-1】补充支付链路日志埋点（加入trace-id、耗时字段）
2. 【T-2】在压测环境重现线上负载并持续30分钟
3. 【T-3】分析火焰图与调用链，定位Top-3 性能瓶颈
4. 【T-4】针对瓶颈实施代码或配置优化并提交MR
5. 【T-5】灰度发布到5%流量并回归验证性能指标

---

【日志继续】
时间戳: 2024-05-21 18:30:00
分段: 任务进度
内容:
- 任务T-1: ✅ 已全部15个服务完成埋点，PR合并
- 任务T-2: ✅ 压测环境复现成功，峰值1200 TPS，日志120 GiB
- 任务T-3: ✅ 瓶颈定位完成：① 订单状态缓存未命中；② 下游风控接口重试浪费；③ JVM GC停顿过长
    """
    resp = check_run_task_is_finished(task_log)
    assert not resp.is_finished

    append_task_log = """
【日志继续】
时间戳: 2024-05-22 11:42:00
分段: 任务进度
内容:
1. 任务T-4 开始。基于 T-3 的瓶颈报告，拆出3个子任务：
   - 4a 缓存层：将订单状态缓存预热脚本提前30分钟运行，命中率提升到92%。
   - 4b 风控接口：在下游重试前插入1 ms 权重退让 + 指数回退，重试次数由6次降到2次。
   - 4c JVM：启用ZGC、缩短停顿，GC Root扫描时间从120 ms降到14 ms。
所有代码/配置改动合并到 release/v2.3.9-hotfix-1，并通过单元测试、静态扫描。
任务T-4 状态: ✅ 已完成

---

【日志继续】
时间戳: 2024-05-22 15:20:00
分段: 任务进度
内容:
任务T-5 启动灰度发布；步骤及观察：
1. 15:20 在 kubernetes 集群中把 v2.3.9-hotfix-1 镜像滚动到5%在线节点（约150台）。
2. 15:25-15:50 观测 Prometheus：
   • order-payment-p99 延迟降至330 ms (目标 ≤500 ms)
   • 资金使用一致性脚本比对，未发现资金差异
   • 卡单率0，告警数为0
3. 16:30 自动关闭5%灰度，Master节点确认无异常，所有节点完成版本升级。
任务T-5 状态: ✅ 已完成
    """

    resp = check_run_task_is_finished(task_log + append_task_log)
    assert resp.is_finished


def test_convert_conversation_to_task_log():
    file_path = logs_dir / "conversation_20251026064326_554669.pkl"
    conversation = pickle.loads(file_path.read_bytes())
    assert isinstance(conversation[-1], ChatCompletionChunk)
    last_chunk = conversation[-1]
    assert last_chunk.choices[0].finish_reason == "stop"
    task_log = run_task_llm.convert_conversation_to_task_log(conversation)
    print(task_log)
