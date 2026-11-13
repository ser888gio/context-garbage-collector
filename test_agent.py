import json
import re
from types import SimpleNamespace as NS

import agent


class FakeAgent:
    """Offline model with real behaviour: lists, reads every file, recalls stubs when it can't find the answer."""

    def __init__(self):
        self.to_read, self.n, self.recalled = None, 0, set()

    def call(self, name, **args):
        self.n += 1
        return [NS(id=f"f{self.n}", function=NS(name=name, arguments=json.dumps(args)))]

    def reply(self, content=None, calls=None, messages=()):
        usage = NS(prompt_tokens=sum(len(json.dumps(m)) for m in messages) // 4, completion_tokens=5)
        return NS(choices=[NS(message=NS(content=content, tool_calls=calls))], usage=usage)

    def __call__(self, messages, tools):
        if self.to_read is None:
            self.to_read = []
            return self.reply(calls=self.call("list_files"), messages=messages)
        if messages[-1]["role"] == "tool" and messages[-1]["content"].startswith("notes/00.txt"):
            self.to_read = messages[-1]["content"].split("\n")
        if self.to_read:
            return self.reply(calls=self.call("read_file", path=self.to_read.pop(0)), messages=messages)
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "access code" not in question:
            return self.reply("DONE", messages=messages)
        name = re.search(r"project (\w+)\?", question).group(1)
        for m in messages:
            hit = re.search(rf"project {name} is (\d+)", m.get("content") or "")
            if hit:
                return self.reply(hit.group(1), messages=messages)
        stubs = [int(i) for m in messages for i in re.findall(r"call recall\((\d+)\)", m.get("content") or "")]
        fresh = [i for i in stubs if i not in self.recalled]
        if fresh:
            self.recalled.add(fresh[0])
            return self.reply(calls=self.call("recall", id=fresh[0]), messages=messages)
        return self.reply("I don't know", messages=messages)


def run(strategy, budget, upfront=False, trigger="threshold:0.7"):
    task = agent.needle_task(seed=7, n_files=10, position=0.5, upfront=upfront)
    row, _ = agent.run(task, "fake/model", strategy, trigger, budget, FakeAgent(), max_turns=200)
    return row


def test_no_gc_with_room_to_spare_answers_correctly_without_recalls():
    row = run("none", budget=100_000)

    assert (row["end"], row["success"], row["recalls"], row["gc_runs"]) == ("answered", True, 0, 0)


def test_no_gc_runs_out_of_memory_when_the_files_exceed_the_budget():
    row = run("none", budget=4000)

    assert (row["end"], row["success"]) == ("oom", False)


def test_mark_sweep_loses_the_needle_when_the_question_comes_late_and_the_agent_recalls_it():
    row = run("mark_sweep", budget=4000)

    assert (row["end"], row["success"]) == ("answered", True)
    assert row["recalls"] > 0


def test_mark_sweep_keeps_the_needle_reachable_when_the_question_is_in_the_task():
    row = run("mark_sweep", budget=4000, upfront=True)

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, 0)


def test_every_turn_trigger_collects_before_every_model_call():
    row = run("lru_mask", budget=100_000, trigger="every_turn")

    assert row["gc_runs"] == row["turns"]


def test_needle_sits_in_the_file_at_the_requested_position():
    task = agent.needle_task(seed=7, n_files=10, position=0.5)

    assert [p for p, body in task["files"].items() if f"is {task['answer']}." in body] == ["notes/05.txt"]


def test_same_seed_builds_the_same_task():
    assert agent.needle_task(seed=3) == agent.needle_task(seed=3)


def test_run_row_records_hints_setting():
    task = agent.needle_task(seed=7, n_files=10, position=0.5)

    row, _ = agent.run(task, "fake/model", "mark_sweep", "threshold:0.7", 4000, FakeAgent(), max_turns=200, hints=True)

    assert row["hints"] is True


def test_oracle_answers_with_zero_recalls_under_a_small_budget():
    row = run("oracle", budget=4000)

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, 0)


def test_common_prefix_of_an_append_only_list_is_the_whole_previous_list():
    assert agent.common_prefix([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}, {"c": 3}]) == (2, 16)


def test_common_prefix_stops_at_an_edited_middle_message():
    assert agent.common_prefix([{"a": 1}, {"b": 2}, {"c": 3}], [{"a": 1}, {"b": 9}, {"c": 3}]) == (1, 8)


def test_common_prefix_is_empty_when_the_first_message_is_edited():
    assert agent.common_prefix([{"a": 1}, {"b": 2}], [{"a": 9}, {"b": 2}]) == (0, 0)


def test_append_only_run_never_breaks_the_cache():
    row = run("none", budget=100_000)

    assert row["cache_breaks"] == 0


def test_mark_sweep_run_breaks_the_cache_and_lowers_prefix_share():
    row = run("mark_sweep", budget=4000)

    assert (row["cache_breaks"], row["stable_prefix_share"]) == (11, 0.373)


class ProviderRejected(Exception):
    pass


class FailsOnCall(FakeAgent):
    def __init__(self, n):
        super().__init__()
        self.limit, self.calls_made = n, 0

    def __call__(self, messages, tools):
        self.calls_made += 1
        if self.calls_made == self.limit:
            raise ProviderRejected("Tool call validation failed: tool_use_failed " + "x" * 400)
        return super().__call__(messages, tools)


class CallsMissingToolOnce(FakeAgent):
    def __init__(self):
        super().__init__()
        self.tried = False

    def __call__(self, messages, tools):
        if not self.tried:
            self.tried = True
            return self.reply(calls=self.call("search", q="x"), messages=messages)
        return super().__call__(messages, tools)


def run_with(complete):
    task = agent.needle_task(seed=7, n_files=10, position=0.5)
    return agent.run(task, "fake/model", "none", "threshold:0.7", 100_000, complete, max_turns=200)[0]


def test_api_error_ends_only_that_run_and_records_the_error():
    row = run_with(FailsOnCall(3))

    assert (row["end"], row["success"], row["turns"]) == ("api_error", False, 3)
    assert row["error"] == "ProviderRejected: Tool call validation failed: tool_use_failed " + "x" * 237


def test_rate_limit_still_propagates_out_of_run():
    def limited(messages, tools):
        raise agent.RateLimited("fake/model")

    try:
        run_with(limited)
    except agent.RateLimited:
        return
    raise AssertionError("RateLimited was swallowed")


def test_normal_run_has_empty_error_and_no_unknown_tool_calls():
    row = run_with(FakeAgent())

    assert (row["error"], row["unknown_tool_calls"]) == ("", 0)


def test_call_to_a_tool_not_in_the_request_is_counted_and_the_run_continues():
    row = run_with(CallsMissingToolOnce())

    assert (row["unknown_tool_calls"], row["end"], row["success"]) == (1, "answered", True)
