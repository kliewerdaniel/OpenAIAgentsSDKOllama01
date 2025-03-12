from __future__ import annotations

import re
from collections.abc import Awaitable
from typing import Any, Literal, Union

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypeVar

from .exceptions import ModelBehaviorError
from .logger import logger
from .tracing import Span, SpanError, get_current_span

T = TypeVar("T")

MaybeAwaitable = Union[Awaitable[T], T]


def transform_string_function_style(name: str) -> str:
    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)

    return name.lower()


def validate_json(json_str: str, type_adapter: TypeAdapter[T], partial: bool) -> T:
    partial_setting: bool | Literal["off", "on", "trailing-strings"] = (
        "trailing-strings" if partial else False
    )
    try:
        validated = type_adapter.validate_json(json_str, experimental_allow_partial=partial_setting)
        return validated
    except ValidationError as e:
        attach_error_to_current_span(
            SpanError(
                message="Invalid JSON provided",
                data={},
            )
        )
        raise ModelBehaviorError(
            f"Invalid JSON when parsing {json_str} for {type_adapter}; {e}"
        ) from e


def attach_error_to_span(span: Span[Any], error: SpanError) -> None:
    span.set_error(error)


def attach_error_to_current_span(error: SpanError) -> None:
    span = get_current_span()
    if span:
        attach_error_to_span(span, error)
    else:
        logger.warning(f"No span to add error {error} to")


async def noop_coroutine() -> None:
    pass
