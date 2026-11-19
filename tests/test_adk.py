import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import Agent  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from cgc.adk import Collector  # noqa: E402
from cgc.agent import SYSTEM  # noqa: E402
from cgc.tasks import needle_task  # noqa: E402


async def run_agent(agent, prompt):
    service = InMemorySessionService()
    runner = Runner(app_name="cgc-test", agent=agent, session_service=service)
    session = await service.create_session(app_name="cgc-test", user_id="u")
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    answer = None
    async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message):
        if event.is_final_response():
            answer = event.content.parts[0].text
    return answer, session.id


@pytest.fixture(scope="module")
def adk_run(scripted):
    agent, env, _ = scripted
    task = needle_task(0, 10, upfront=True)

    def list_files() -> str:
        """List all note files."""
        return "\n".join(task["files"])

    def read_file(path: str) -> str:
        """Read one note file."""
        return task["files"][path]

    collector = Collector(strategy="truncate", budget=4000, hints="keys")
    adk_agent = Agent(name="notes", instruction=SYSTEM,
                      model=LiteLlm(model="openai/scripted-agent", api_base=env["OPENAI_BASE_URL"], api_key="e2e"),
                      before_model_callback=collector.before_model, tools=[list_files, read_file, *collector.adk_tools])
    start = len(agent.seen)
    agent.unique_ids = True
    try:
        answer, session_id = asyncio.run(run_agent(adk_agent, task["prompt"]))
    finally:
        agent.unique_ids = False
    return answer, collector.context(session_id), agent.seen[start:]


def texts(requests):
    return [m["content"] for messages in requests for m in messages if isinstance(m.get("content"), str)]


def test_the_model_received_stubs_for_collected_notes(adk_run):
    _, _, requests = adk_run

    assert any(t.startswith("[collected") and "call recall(" in t for t in texts(requests))


def test_a_recall_through_the_adk_tool_returned_the_original_note(adk_run):
    _, _, requests = adk_run

    recalled = [t for t in texts(requests[-1:]) if "[recalled msg" in t]
    assert len(recalled) == 1 and "The access code for project ORBITAL is 5242" in recalled[0]


def test_the_agent_answered_correctly_and_the_session_counted_the_recall(adk_run):
    answer, ctx, _ = adk_run

    assert (answer, ctx.stats["recalls"], ctx.stats["searches"]) == ("5242", 1, 1)
