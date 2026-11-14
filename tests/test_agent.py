import json
import re
from types import SimpleNamespace as NS

import pytest

from cgc import agent
from cgc import tasks


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

    def find(self, messages, name):
        for m in messages:
            hit = re.search(rf"project {name} is (\d+)", m.get("content") or "")
            if hit:
                return hit.group(1)

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
        if (found := self.find(messages, name)) is not None:
            return self.reply(found, messages=messages)
        stubs = [int(i) for m in messages for i in re.findall(r"call recall\((\d+)\)", m.get("content") or "")]
        fresh = [i for i in stubs if i not in self.recalled]
        if fresh:
            self.recalled.add(fresh[0])
            return self.reply(calls=self.call("recall", id=fresh[0]), messages=messages)
        return self.reply("I don't know", messages=messages)


class FakeHopAgent(FakeAgent):
    """Follows 'same as the code for project X' links before reporting a value."""

    def find(self, messages, name):
        for _ in range(3):
            for m in messages:
                content = m.get("content") or ""
                if hit := re.search(rf"project {name} is (\d+)", content):
                    return hit.group(1)
                if hop := re.search(rf"project {name} is the same as the code for project (\w+)", content):
                    name = hop.group(1)
                    break
            else:
                return None
        return None


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


def test_run_row_records_tag():
    task = agent.needle_task(seed=7, n_files=10, position=0.5)

    row, _ = agent.run(task, "fake/model", "none", "threshold:0.7", 100_000, FakeAgent(), max_turns=200, tag="sweep2")

    assert row["tag"] == "sweep2"


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


class Summarizing(FakeAgent):
    def __call__(self, messages, tools):
        if not tools:
            return self.reply("SUM", messages=messages)
        return super().__call__(messages, tools)


def test_hybrid_run_records_the_summary_calls_it_made():
    task = agent.needle_task(seed=7, n_files=10, position=0.5)

    row, _ = agent.run(task, "fake/model", "hybrid", "threshold:0.7", 4000, Summarizing(), max_turns=200)

    assert row["summary_calls"] == 3


class Hoarder(FakeAgent):
    """Reads everything and never forgets; strips the [msg N] prefix manual mode adds."""

    def call(self, name, **args):
        self.last = name
        return super().call(name, **args)

    def __call__(self, messages, tools):
        return super().__call__(self.plain(messages), tools)

    @staticmethod
    def plain(messages):
        return [dict(m, content=re.sub(r"^\[msg \d+\] ", "", m["content"])) if m["role"] == "tool" else m
                for m in messages if not (m["content"] or "").startswith("[memory]")]


class Forgetful(Hoarder):
    def __init__(self):
        super().__init__()
        self.facts, self.reads = {}, []

    def __call__(self, messages, tools):
        messages = [m for m in messages if not (m["content"] or "").startswith("[memory]")]
        last = messages[-1]
        if last["role"] == "tool" and getattr(self, "last", None) == "read_file":
            self.facts.update(re.findall(r"project (\w+) is (\d+)", last["content"]))
            old, self.reads = self.reads, [int(re.match(r"\[msg (\d+)\]", last["content"]).group(1))]
            if old:
                return self.reply(calls=self.call("forget", ids=old), messages=messages)
        if self.to_read == [] and self.facts:
            name = re.search(r"project (\w+)\?", messages[1]["content"]).group(1)
            return self.reply(self.facts[name], messages=messages)
        return super().__call__(messages, tools)


def run_manual(model, budget=4000, pressure=False):
    task = agent.needle_task(seed=7, n_files=10, position=0.5, upfront=True)
    return agent.run(task, "fake/model", "manual", "threshold:0.7", budget, model, max_turns=200, pressure=pressure)[0]


def test_manual_model_that_forgets_old_file_reads_answers_within_a_small_budget():
    row = run_manual(Forgetful())

    assert (row["end"], row["success"], row["forgets"]) == ("answered", True, 9)


def test_manual_model_that_never_forgets_runs_out_of_memory():
    row = run_manual(Hoarder())

    assert (row["end"], row["success"], row["forgets"]) == ("oom", False, 0)


def test_pressure_note_reports_usage_and_collected_count_of_a_small_heap():
    from cgc import collectors as C
    heap = [C.Obj(0, {"role": "user", "content": "x" * 400}, 0, root=True),
            C.Obj(1, {"role": "tool", "content": "y"}, 0, dead=True)]

    assert agent.pressure_note(heap, 200)["content"] == "[memory] context 57% of budget (115/200 tokens), 1 messages collected."


def test_pressure_note_is_not_part_of_the_cached_prefix():
    off = run_manual(Forgetful(), budget=8000, pressure=False)
    on = run_manual(Forgetful(), budget=8000, pressure=True)

    assert (off["stable_prefix_share"], on["stable_prefix_share"]) == (0.481, 0.481)


def run_hop(strategy, budget, upfront):
    task = tasks.multihop_task(seed=7, n_files=10, position=0.5, upfront=upfront)
    return agent.run(task, "fake/model", strategy, "threshold:0.7", budget, FakeHopAgent(), max_turns=200)[0]


def test_multihop_puts_the_link_and_the_value_in_different_files():
    task = tasks.multihop_task(seed=7, n_files=10, position=0.5)

    assert [p for p, body in task["files"].items() if "is the same as the code for project VEGA" in body] == ["notes/05.txt"]
    assert [p for p, body in task["files"].items() if "project VEGA is 7468" in body] == ["notes/09.txt"]
    assert (task["id"], task["family"], task["answer"]) == ("multihop-p0.5-late-n10-s7", "multihop", "7468")


def test_poison_puts_the_old_code_early_and_the_correction_later():
    task = tasks.poison_task(seed=7, n_files=10)

    assert [p for p, body in task["files"].items() if "project ORION is 7468." in body] == ["notes/02.txt"]
    assert [p for p, body in task["files"].items() if "ORION is now 1791; the old code 7468 is revoked" in body] == ["notes/03.txt"]
    assert (task["id"], task["answer"], task["wrong"]) == ("poison-p0.2-late-n10-s7", "1791", "7468")


def test_task_builders_are_deterministic():
    assert tasks.multihop_task(seed=3) == tasks.multihop_task(seed=3)
    assert tasks.poison_task(seed=3) == tasks.poison_task(seed=3)


def test_mark_sweep_follows_the_reference_chain_on_multihop_with_zero_recalls():
    row = run_hop("mark_sweep", budget=4000, upfront=True)

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, 0)


def test_mark_sweep_recalls_eleven_times_on_multihop_when_the_question_comes_late():
    row = run_hop("mark_sweep", budget=4000, upfront=False)

    assert (row["end"], row["success"], row["recalls"]) == ("answered", True, 11)


@pytest.mark.parametrize("strategy", ["truncate", "lru_mask"])
def test_truncate_and_lru_fail_multihop_even_after_eleven_recalls(strategy):
    row = run_hop(strategy, budget=4000, upfront=True)

    assert (row["success"], row["recalls"], row["answer"]) == (False, 11, "I don't know")


def answer_poison(answer):
    task = tasks.poison_task(seed=7, n_files=10, upfront=True)

    def complete(messages, tools):
        return FakeAgent().reply(answer, messages=messages)
    return agent.run(task, "fake/model", "none", "threshold:0.7", 100_000, complete)[0]["success"]


@pytest.mark.parametrize("answer,success", [("1791", True), ("7468", False), ("changed from 7468 to 1791", False)])
def test_poison_scoring_requires_the_new_code_and_rejects_the_revoked_one(answer, success):
    assert answer_poison(answer) is success


def run_row(strategy, trigger, hints=False, upfront=False):
    task = agent.needle_task(seed=7, n_files=10, position=0.5, upfront=upfront)
    return agent.run(task, "fake/model", strategy, trigger, 4000, FakeAgent(), max_turns=200, hints=hints)[0]


def test_preview_stub_run_records_the_mode_and_still_succeeds():
    row = run_row("mark_sweep", "threshold:0.7", hints="preview")

    assert (row["hints"], row["success"], row["cache_breaks"]) == ("preview", True, 10)


def test_numbers_stub_run_records_the_mode_and_still_succeeds():
    row = run_row("mark_sweep", "threshold:0.7", hints="numbers")

    assert (row["hints"], row["success"], row["cache_breaks"]) == ("numbers", True, 10)


def test_hysteresis_run_collects_the_reachable_needle_that_plain_threshold_keeps():
    plain = run_row("mark_sweep", "threshold:0.7", hints=True, upfront=True)
    low = run_row("mark_sweep", "threshold:0.7:0.4", hints=True, upfront=True)

    assert [(r["success"], r["cache_breaks"], r["recalls"]) for r in (plain, low)] == [(True, 5, 0), (True, 10, 6)]


def test_h2o_run_succeeds_by_recalling_what_it_collected():
    row = run_row("h2o", "threshold:0.7")

    assert (row["success"], row["cache_breaks"], row["recalls"]) == (True, 13, 6)


def test_mark_sweep_chain_answers_multihop_with_zero_recalls_when_the_question_is_in_the_task():
    row = run_hop("mark_sweep_chain", budget=4000, upfront=True)

    assert (row["end"], row["success"], row["recalls"], row["tokens_freed"]) == ("answered", True, 0, 6200)


def test_mark_sweep_chain_recalls_ten_times_on_multihop_when_the_question_comes_late():
    row = run_hop("mark_sweep_chain", budget=4000, upfront=False)

    assert (row["end"], row["success"], row["recalls"], row["tokens_freed"]) == ("answered", True, 10, 13972)
