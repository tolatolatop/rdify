import anyio
import logging
from uuid import uuid4
from fastapi import Request

class CancelScope:

    def __init__(self, **kwargs):
        self.request = kwargs.get("context", {}).get("request")
        self.id = str(uuid4())[:4]
        self.logger = logging.getLogger(f"rdify.cancel_scope.id_{self.id}")
        self.cancel_scope = None
        self._ready = False

    def __enter__(self):
        self.logger.debug(f"CancelScope __enter__: {self.id}")
        self.cancel_scope = anyio.CancelScope()
        self._ready = True
        if not isinstance(self.cancel_scope, anyio.CancelScope):
            self.logger.debug(f"CancelScope wait: {self.id} cancel_scope is not a CancelScope")
            self._ready = False
        if not hasattr(self.request, "is_disconnected"):
            self.logger.debug(f"CancelScope wait: {self.id} request is not a Request")
            self._ready = False
        if not callable(self.request.is_disconnected):
            self.logger.debug(f"CancelScope wait: {self.id} request.is_disconnected is not callable")
            self._ready = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(exc_type, anyio.get_cancelled_exc_class()):
            self.logger.debug(f"CancelScope __exit__: {self.id} cancelled")
            return True
        return False

    def cancel(self):
        self.logger.debug(f"CancelScope cancel: {self.id}")
        if isinstance(self.cancel_scope, anyio.CancelScope):
            self.cancel_scope.cancel()

    async def wait(self):
        if not self._ready:
            self.logger.debug(f"CancelScope wait: {self.id} not ready")
            return True
        is_disconnected = await self.request.is_disconnected()
        self.logger.debug(f"CancelScope wait: {self.id} is_disconnected: {is_disconnected}")
        if is_disconnected:
            self.cancel_scope.cancel()
            return True
        return True
