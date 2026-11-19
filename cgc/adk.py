import json

from google.adk.tools.tool_context import ToolContext
from google.genai import types

from cgc.context import DEFAULTS, Context


def tool_text(fr):
    r = fr.response or {}
    return r["result"] if list(r) == ["result"] and isinstance(r["result"], str) else json.dumps(r)


def to_messages(contents):
    out, fcs, frs = [], {}, {}
    for c in contents:
        parts = c.parts or []
        if c.role == "model":
            text = "".join(p.text for p in parts if p.text and not p.thought)
            calls = [p for p in parts if p.function_call]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = [{"id": p.function_call.id, "type": "function",
                                      "function": {"name": p.function_call.name,
                                                   "arguments": json.dumps(p.function_call.args or {})}} for p in calls]
            out.append(msg)
            fcs |= {p.function_call.id: p for p in calls}
        else:
            for p in parts:
                if p.function_response:
                    frs[p.function_response.id] = p
                    out.append({"role": "tool", "tool_call_id": p.function_response.id, "content": tool_text(p.function_response)})
                elif p.text:
                    out.append({"role": "user", "content": p.text})
    return out, fcs, frs


def to_contents(messages, fcs, frs):
    out, last_was_tool = [], False
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            orig = frs[m["tool_call_id"]]
            same = tool_text(orig.function_response) == m["content"]
            part = orig if same else types.Part(function_response=types.FunctionResponse(
                id=orig.function_response.id, name=orig.function_response.name, response={"result": m["content"]}))
            if last_was_tool:
                out[-1].parts.append(part)
            else:
                out.append(types.Content(role="user", parts=[part]))
        elif m["role"] == "assistant":
            parts = [types.Part(text=m["content"])] if m.get("content") else []
            parts += [fcs[c["id"]] for c in m.get("tool_calls") or []]
            out.append(types.Content(role="model", parts=parts))
        else:
            out.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        last_was_tool = m["role"] == "tool"
    return out


def system_text(instruction):
    if isinstance(instruction, str) or instruction is None:
        return instruction
    return "".join(p.text or "" for p in instruction.parts)


class Collector:
    def __init__(self, **options):
        self.options = options
        # ponytail: in-memory, one process only; multi-instance deployments need the Context state in session.state.
        self.sessions = {}

        def recall(id: int, tool_context: ToolContext) -> str:
            """Return the original content of a collected message, given the id in its [collected: msg N ...] stub."""
            return self.context(tool_context.session.id).call("recall", {"id": id})

        def search(query: str, tool_context: ToolContext) -> str:
            """Search the text of collected messages without restoring them. Returns matching sentences with their message ids."""
            return self.context(tool_context.session.id).call("search", {"query": query})

        self.adk_tools = [recall, search][:1 + options.get("search", DEFAULTS["search"])]

    def context(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = Context(**self.options)
        return self.sessions[session_id]

    def before_model(self, callback_context, llm_request):
        messages, fcs, frs = to_messages(llm_request.contents)
        system = system_text(llm_request.config.system_instruction)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        prepared = self.context(callback_context.session.id).prepare(messages)
        llm_request.contents = to_contents(prepared, fcs, frs)
        return None
