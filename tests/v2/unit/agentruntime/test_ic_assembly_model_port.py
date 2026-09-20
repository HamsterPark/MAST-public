"""``ic_assembly._as_port``: an explicit LangChain chat model must be WRAPPED.

``ChatModelPort`` is a structural Protocol (``invoke`` + ``stream``) and every
LangChain ``BaseChatModel`` has both, so ``isinstance(model, ChatModelPort)`` was
true for raw LangChain models and the loop fed them a ``ModelRequest``:
``ValueError: Invalid input type <class 'ModelRequest'>`` (2026-08-28, STM-Bench
driver, the first caller to pass ``model=`` explicitly).
"""
from __future__ import annotations


class _LangChainLike:
    """Shape of a LangChain chat model: invoke/stream AND bind_tools."""

    def invoke(self, x):
        return x

    def stream(self, x):
        yield x

    def bind_tools(self, tools, **kw):
        return self


class _PortLike:
    """A genuine port: invoke/stream only, speaks ModelRequest."""

    def invoke(self, request):
        return request

    def stream(self, request):
        yield ""


def test_langchain_models_are_wrapped_and_ports_pass_through():
    from mast.agentruntime.ic_assembly import _as_port
    from mast.agentruntime.model import LangChainModelPort

    lc = _LangChainLike()
    assert isinstance(_as_port(lc), LangChainModelPort)
    port = _PortLike()
    assert _as_port(port) is port
    wrapped = LangChainModelPort(lc)
    assert _as_port(wrapped) is wrapped
