import threading
import queue
from typing import Callable, Iterator, TypeVar, Optional, Any, Generic
import contextlib
import traceback
import sys
import asyncio


T = TypeVar("T")


class _EndOfStream:
    pass


class _ErrorEnvelope:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.traceback_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class ThreadQueueBridge(Generic[T]):
    """
    在线程中运行阻塞/同步迭代器，通过队列与异步消费者通信。

    用法：
        bridge = ThreadQueueBridge(max_queue_size=100)
        async for item in bridge.run(lambda: some_blocking_iter(...)):
            ...
    """

    def __init__(self, max_queue_size: int = 100):
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _producer(self, iter_fn: Callable[[], Iterator[T]]):
        try:
            iterator = iter_fn()
            for item in iterator:
                if self._stop_event.is_set():
                    break
                self._queue.put(item)
        except BaseException as exc:  # noqa: BLE001
            # 把异常封装并投递到队列，交由异步侧抛出
            self._queue.put(_ErrorEnvelope(exc))
        finally:
            # 结束信号
            self._queue.put(_EndOfStream())

    async def run(self, iter_fn: Callable[[], Iterator[T]]):
        """
        启动后台线程执行 iter_fn()，返回可异步迭代的生成器。
        """
        if self._thread is not None:
            raise RuntimeError("ThreadQueueBridge can only be used once per instance")

        self._thread = threading.Thread(target=self._producer, args=(iter_fn,), daemon=True)
        self._thread.start()

        loop = asyncio.get_running_loop()

        while True:
            item = await loop.run_in_executor(None, self._queue.get)
            if isinstance(item, _EndOfStream):
                break
            if isinstance(item, _ErrorEnvelope):
                # 在异步侧重新抛出原异常（带回溯文本作为信息）
                with contextlib.suppress(Exception):
                    # 帮助在日志中打印原始回溯
                    sys.stderr.write(item.traceback_str)
                raise item.exc
            yield item  # type: ignore[misc]

        # 等待线程结束（尽量不阻塞事件循环）
        if self._thread.is_alive():
            await loop.run_in_executor(None, self._thread.join)

    def cancel(self):
        """请求结束生产。注意：若底层迭代器自身不可中断，则要依赖其自然结束。"""
        self._stop_event.set()


async def run_blocking_iter_in_thread(iter_fn: Callable[[], Iterator[T]], *, max_queue_size: int = 100):
    """
    便捷函数：在后台线程运行一个阻塞/同步迭代器，返回异步可迭代对象。
    """
    bridge: ThreadQueueBridge[T] = ThreadQueueBridge(max_queue_size=max_queue_size)
    async for item in bridge.run(iter_fn):
        yield item


