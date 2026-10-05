import json
import re
import sqlite3
from typing import TypedDict

import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

import tools

llm = ChatOllama(model="llama3.1", temperature=0, num_ctx=8192)
MAX_ATTEMPTS = 3


class State(TypedDict, total=False):
    df: pd.DataFrame
    anomalies: list
    channel_stats: dict
    facts: str
    brief: str
    attempts: int
    issues: list
    approved: bool


def load_data(state):
    return {"df": tools.load()}


def find_anomalies(state):
    return {"anomalies": tools.detect_anomalies(state["df"])}


def test_channels(state):
    return {"channel_stats": tools.test_channels(state["df"])}


def write_brief(state):
    keep = ["channel", "spend", "revenue", "mean_daily_roas", "roas_ci_low", "roas_ci_high", "losing_money"]
    channels = [{k: c[k] for k in keep} for c in state["channel_stats"]["channels"]]
    facts = json.dumps({"anomalies": state["anomalies"], "channels": channels})
    prompt = (
        "DATA:\n" + facts + "\n\n"
        "TASK: You are a marketing analytics auditor. Using ONLY the data above, write a short budget brief with: "
        "1) channels where losing_money is true (these lose money with statistical significance), "
        "2) recent cost-per-click anomalies (every anomaly listed is a spike far ABOVE that channel's normal cost per click), "
        "3) a recommended budget reallocation: move budget away from channels where losing_money is true, "
        "do not move budget into channels with recent anomalies, and favor channels with the highest mean_daily_roas. "
        "Do not describe the data format. Do not invent numbers, percentages, or dollar amounts that are not in the data."
    )
    if state.get("issues"):
        prompt += f"\n\nYour previous draft used numbers that are not in the data: {state['issues']}. Rewrite it without them."
    return {"brief": llm.invoke(prompt).content, "facts": facts, "attempts": state.get("attempts", 0) + 1}


def check_brief(state):
    """Reject any number in the brief that does not appear in the data."""
    allowed = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", state["facts"])} | {1, 2, 3, 95}
    text = re.sub(r"(?<=\d),(?=\d)", "", state["brief"])
    issues = sorted({n for n in re.findall(r"\d+(?:\.\d+)?", text) if float(n) not in allowed})
    return {"issues": issues}


def route_after_check(state):
    if state["issues"] and state["attempts"] < MAX_ATTEMPTS:
        return "write_brief"
    return "human_review"


def human_review(state):
    print("\n" + state["brief"])
    if state["issues"]:
        print(f"\nWARNING: numbers not found in the data: {state['issues']}")
    return {"approved": input("\nApprove and save? (y/n): ").strip().lower() == "y"}


def save_outputs(state):
    with open("audit_brief.md", "w") as f:
        f.write(state["brief"])
    pd.DataFrame(state["channel_stats"]["channels"]).to_csv("channel_summary.csv", index=False)
    if state["anomalies"]:
        with sqlite3.connect("marketing.db") as con:
            pd.DataFrame(state["anomalies"]).to_sql("anomalies", con, if_exists="replace", index=False)
    print("Saved audit_brief.md, channel_summary.csv, and anomalies table.")
    return {}


graph = StateGraph(State)
for name, fn in [("load_data", load_data), ("find_anomalies", find_anomalies), ("test_channels", test_channels),
                 ("write_brief", write_brief), ("check_brief", check_brief),
                 ("human_review", human_review), ("save_outputs", save_outputs)]:
    graph.add_node(name, fn)

graph.add_edge(START, "load_data")
graph.add_edge("load_data", "find_anomalies")
graph.add_edge("find_anomalies", "test_channels")
graph.add_edge("test_channels", "write_brief")
graph.add_edge("write_brief", "check_brief")
graph.add_conditional_edges("check_brief", route_after_check)
graph.add_conditional_edges("human_review", lambda s: "save_outputs" if s["approved"] else END)
graph.add_edge("save_outputs", END)

app = graph.compile()

if __name__ == "__main__":
    app.invoke({})
