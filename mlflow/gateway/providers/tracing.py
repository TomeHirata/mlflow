from typing import Any, AsyncIterable

from mlflow.gateway.providers.base import BaseProvider, PassthroughAction
from mlflow.gateway.schemas import chat, completions, embeddings


class TracingProviderWrapper(BaseProvider):
    """
    A wrapper provider that adds MLflow tracing spans to all provider method calls.

    This wrapper automatically instruments any provider (including FallbackProvider
    and TrafficRouteProvider) with tracing spans for each method invocation.

    Usage:
        provider = TracingProviderWrapper(original_provider)
        result = await provider.chat(payload)  # Automatically traced

    The wrapper creates spans with:
        - Provider name and model information
        - Method name being called
        - Success/error status
        - Error details on failure
    """

    NAME: str = "TracingWrapper"

    def __init__(self, provider: BaseProvider):
        self._provider = provider
        # Expose underlying provider attributes for compatibility
        if hasattr(provider, "config"):
            self.config = provider.config

    @property
    def wrapped_provider(self) -> BaseProvider:
        """Access the underlying wrapped provider."""
        return self._provider

    def _get_span_name(self) -> str:
        """Generate span name based on wrapped provider."""
        provider_name = getattr(self._provider, "NAME", type(self._provider).__name__)
        model_name = ""
        if hasattr(self._provider, "config") and hasattr(self._provider.config, "model"):
            model_name = getattr(self._provider.config.model, "name", "")

        span_name = f"provider/{provider_name}"
        if model_name:
            span_name = f"{span_name}/{model_name}"
        return span_name

    def _get_provider_attributes(self) -> dict[str, str]:
        """Get provider attributes for span."""
        attrs = {
            "provider": getattr(self._provider, "NAME", type(self._provider).__name__),
        }
        if hasattr(self._provider, "config") and hasattr(self._provider.config, "model"):
            if model_name := getattr(self._provider.config.model, "name", ""):
                attrs["model"] = model_name
        return attrs

    async def _trace_method(self, method_name: str, method, *args, **kwargs):
        """Execute a method with tracing span."""
        import mlflow

        active_span = mlflow.get_current_active_span()
        if active_span is None:
            return await method(*args, **kwargs)

        span_name = self._get_span_name()
        with mlflow.start_span(name=span_name) as span:
            for key, value in self._get_provider_attributes().items():
                span.set_attribute(key, value)
            span.set_attribute("method", method_name)

            try:
                result = await method(*args, **kwargs)
                span.set_status("OK")
                return result
            except Exception as e:
                span.set_status("ERROR")
                span.set_attribute("error", str(e))
                raise

    async def _trace_stream_method(self, method_name: str, method, *args, **kwargs):
        """Execute a streaming method with tracing span."""
        import mlflow
        from mlflow.tracing.fluent import start_span_no_context

        active_span = mlflow.get_current_active_span()
        if active_span is None:
            async for chunk in method(*args, **kwargs):
                yield chunk
            return

        span_name = self._get_span_name()
        # Use start_span_no_context to get a LiveSpan that can be manually ended
        span = start_span_no_context(
            name=span_name,
            parent_span=active_span,
            attributes={
                **self._get_provider_attributes(),
                "method": method_name,
                "streaming": True,
            },
        )

        try:
            last_chunk = None
            async for chunk in method(*args, **kwargs):
                last_chunk = chunk
                yield chunk

            # Extract usage from the final chunk if available (OpenAI includes this
            # when stream_options.include_usage=true)
            if last_chunk is not None and hasattr(last_chunk, "usage") and last_chunk.usage:
                usage = last_chunk.usage
                if hasattr(usage, "prompt_tokens") and usage.prompt_tokens is not None:
                    span.set_attribute("prompt_tokens", usage.prompt_tokens)
                if hasattr(usage, "completion_tokens") and usage.completion_tokens is not None:
                    span.set_attribute("completion_tokens", usage.completion_tokens)
                if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
                    span.set_attribute("total_tokens", usage.total_tokens)

            span.set_status("OK")
            span.end()
        except Exception as e:
            span.set_status("ERROR")
            span.set_attribute("error", str(e))
            span.end()
            raise

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        async for chunk in self._trace_stream_method(
            "chat_stream", self._provider.chat_stream, payload
        ):
            yield chunk

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        return await self._trace_method("chat", self._provider.chat, payload)

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        async for chunk in self._trace_stream_method(
            "completions_stream", self._provider.completions_stream, payload
        ):
            yield chunk

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        return await self._trace_method("completions", self._provider.completions, payload)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return await self._trace_method("embeddings", self._provider.embeddings, payload)

    async def passthrough(
        self,
        action: PassthroughAction,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | AsyncIterable[bytes]:
        return await self._trace_method(
            "passthrough", self._provider.passthrough, action, payload, headers
        )
