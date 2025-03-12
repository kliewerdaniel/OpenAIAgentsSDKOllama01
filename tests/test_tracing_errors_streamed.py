from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from typing_extensions import TypedDict

from agents import (
    Agent,
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
)

from .fake_model import FakeModel
from .test_responses import (
    get_final_output_message,
    get_function_tool,
    get_function_tool_call,
    get_handoff_tool_call,
    get_text_message,
)
from .testing_processor import fetch_ordered_spans, fetch_traces


@pytest.mark.asyncio
async def test_single_turn_model_error():
    model = FakeModel(tracing_enabled=True)
    model.set_next_output(ValueError("test error"))

    agent = Agent(
        name="test_agent",
        model=model,
    )
    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 2, f"should have agent and generation spans, got {len(spans)}"

    generation_span = spans[1]
    assert isinstance(generation_span.span_data, GenerationSpanData)
    assert generation_span.error, "should have error"


@pytest.mark.asyncio
async def test_multi_turn_no_handoffs():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a message and tool call
            [get_text_message("a_message"), get_function_tool_call("foo", json.dumps({"a": "b"}))],
            # Second turn: error
            ValueError("test error"),
            # Third turn: text message
            [get_text_message("done")],
        ]
    )

    with pytest.raises(ValueError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 4, (
        f"should have agent, generation, tool, generation, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )

    last_generation_span = [x for x in spans if isinstance(x.span_data, GenerationSpanData)][-1]
    assert last_generation_span.error, "should have error"


@pytest.mark.asyncio
async def test_tool_call_error():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test_agent",
        model=model,
        tools=[get_function_tool("foo", "tool_result", hide_errors=True)],
    )

    model.set_next_output(
        [get_text_message("a_message"), get_function_tool_call("foo", "bad_json")],
    )

    with pytest.raises(ModelBehaviorError):
        result = Runner.run_streamed(agent, input="first_test")
        async for _ in result.stream_events():
            pass

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 3, (
        f"should have agent, generation, tool spans, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )

    function_span = [x for x in spans if isinstance(x.span_data, FunctionSpanData)][0]
    assert function_span.error, "should have error"


@pytest.mark.asyncio
async def test_multiple_handoff_doesnt_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
    )
    agent_2 = Agent(
        name="test",
        model=model,
    )
    agent_3 = Agent(
        name="test",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: text message
            [get_text_message("done")],
        ]
    )
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.last_agent == agent_1, "should have picked first handoff"

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 7, (
        f"should have 2 agent, 1 function, 3 generation, 1 handoff, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )


class Foo(TypedDict):
    bar: str


@pytest.mark.asyncio
async def test_multiple_final_output_no_error():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test",
        model=model,
        output_type=Foo,
    )

    model.set_next_output(
        [
            get_final_output_message(json.dumps(Foo(bar="baz"))),
            get_final_output_message(json.dumps(Foo(bar="abc"))),
        ]
    )

    result = Runner.run_streamed(agent_1, input="user_message")
    async for _ in result.stream_events():
        pass

    assert isinstance(result.final_output, dict)
    assert result.final_output["bar"] == "abc"

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 2, (
        f"should have 1 agent, 1 generation, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )


@pytest.mark.asyncio
async def test_handoffs_lead_to_correct_agent_spans():
    model = FakeModel(tracing_enabled=True)

    agent_1 = Agent(
        name="test_agent_1",
        model=model,
        tools=[get_function_tool("some_function", "result")],
    )
    agent_2 = Agent(
        name="test_agent_2",
        model=model,
        handoffs=[agent_1],
        tools=[get_function_tool("some_function", "result")],
    )
    agent_3 = Agent(
        name="test_agent_3",
        model=model,
        handoffs=[agent_1, agent_2],
        tools=[get_function_tool("some_function", "result")],
    )

    agent_1.handoffs.append(agent_3)

    model.add_multiple_turn_outputs(
        [
            # First turn: a tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Second turn: a message and 2 handoff
            [
                get_text_message("a_message"),
                get_handoff_tool_call(agent_1),
                get_handoff_tool_call(agent_2),
            ],
            # Third turn: tool call
            [get_function_tool_call("some_function", json.dumps({"a": "b"}))],
            # Fourth turn: handoff
            [get_handoff_tool_call(agent_3)],
            # Fifth turn: text message
            [get_text_message("done")],
        ]
    )
    result = Runner.run_streamed(agent_3, input="user_message")
    async for _ in result.stream_events():
        pass

    assert result.last_agent == agent_3, (
        f"should have ended on the third agent, got {result.last_agent.name}"
    )

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 12, (
        f"should have 3 agents, 2 function, 5 generation, 2 handoff, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )


@pytest.mark.asyncio
async def test_max_turns_exceeded():
    model = FakeModel(tracing_enabled=True)

    agent = Agent(
        name="test",
        model=model,
        output_type=Foo,
        tools=[get_function_tool("foo", "result")],
    )

    model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
            [get_function_tool_call("foo")],
        ]
    )

    with pytest.raises(MaxTurnsExceeded):
        result = Runner.run_streamed(agent, input="user_message", max_turns=2)
        async for _ in result.stream_events():
            pass

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 5, (
        f"should have 1 agent, 2 generations, 2 function calls, got "
        f"{len(spans)} with data: {[x.span_data for x in spans]}"
    )

    agent_span = [x for x in spans if isinstance(x.span_data, AgentSpanData)][-1]
    assert agent_span.error, "last agent should have error"


def input_guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_input_guardrail_error():
    model = FakeModel()

    agent = Agent(
        name="test",
        model=model,
        input_guardrails=[InputGuardrail(guardrail_function=input_guardrail_function)],
    )
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(InputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass

    await asyncio.sleep(1)

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 2, (
        f"should have 1 agent, 1 guardrail, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )

    agent_span = [x for x in spans if isinstance(x.span_data, AgentSpanData)][-1]
    assert agent_span.error, "last agent should have error"


def output_guardrail_function(
    context: RunContextWrapper[Any], agent: Agent[Any], agent_output: Any
) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=True,
    )


@pytest.mark.asyncio
async def test_output_guardrail_error():
    model = FakeModel()

    agent = Agent(
        name="test",
        model=model,
        output_guardrails=[OutputGuardrail(guardrail_function=output_guardrail_function)],
    )
    model.set_next_output([get_text_message("some_message")])

    with pytest.raises(OutputGuardrailTripwireTriggered):
        result = Runner.run_streamed(agent, input="user_message")
        async for _ in result.stream_events():
            pass

    await asyncio.sleep(1)

    traces = fetch_traces()
    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"

    spans = fetch_ordered_spans()
    assert len(spans) == 2, (
        f"should have 1 agent, 1 guardrail, got {len(spans)} with data: "
        f"{[x.span_data for x in spans]}"
    )

    agent_span = [x for x in spans if isinstance(x.span_data, AgentSpanData)][-1]
    assert agent_span.error, "last agent should have error"
