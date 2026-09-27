"""
Defines the three-agent code audit crew.

Agents
------
critic     - Looks for concrete problems: bugs, code smells, risk areas.
defender   - Explains and justifies design decisions, adding context
             a pure static-analysis pass would miss.
arbiter    - Reads both perspectives and produces the final,
             prioritized verdict.

The agents run sequentially: critic -> defender -> arbiter, with the
arbiter's task receiving both previous outputs as context. This
mirrors a human code-review discussion rather than a single
one-shot "review this code" prompt.
"""

from __future__ import annotations

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM


def _build_llm() -> LLM:
    """
    Builds the LLM used by every agent in the crew.

    Defaults to a local Ollama model, matching the setup already
    used in the TCC (DeepSeek-R1 / Gemma via Ollama). Override with
    OLLAMA_MODEL / OLLAMA_BASE_URL in .env, or swap the provider
    entirely (e.g. "openai/gpt-4o") if OPENAI_API_KEY is set.
    """

    model = os.getenv("AUDIT_LLM_MODEL", "ollama/deepseek-r1:14b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if model.startswith("ollama/"):
        return LLM(model=model, base_url=base_url, temperature=0.2)

    # Any other LiteLLM-supported provider (openai/..., anthropic/...)
    return LLM(model=model, temperature=0.2)


def build_agents(llm: LLM) -> tuple[Agent, Agent, Agent]:
    critic = Agent(
        role="Senior Code Reviewer",
        goal=(
            "Find concrete, specific problems in the given source file: "
            "bugs, code smells, risky assumptions, missing validation, "
            "and maintainability issues. Be blunt but precise — cite "
            "line-level behavior, not vague impressions."
        ),
        backstory=(
            "You have reviewed thousands of pull requests in production "
            "ML systems. You are known for catching the one edge case "
            "everyone else missed. You do not soften criticism, but you "
            "never invent a problem that isn't actually in the code."
        ),
        llm=llm,
        verbose=True,
    )

    defender = Agent(
        role="Software Architect",
        goal=(
            "Explain the reasoning that plausibly justifies the file's "
            "design decisions, including the ones the Critic flagged. "
            "Where a criticism is fair, say so plainly instead of "
            "defending it anyway — your job is honest context, not "
            "unconditional defense."
        ),
        backstory=(
            "You design data and ML systems that run in production. "
            "You know that some 'obviously wrong' patterns exist for a "
            "reason (a deadline, a deliberate trade-off, a constraint "
            "from an upstream system), and you can usually tell the "
            "difference between a trade-off and a real oversight."
        ),
        llm=llm,
        verbose=True,
    )

    arbiter = Agent(
        role="Tech Lead",
        goal=(
            "Read the Critic's findings and the Architect's context, "
            "then produce a final, prioritized verdict: which issues "
            "are real and should be fixed, which are acceptable "
            "trade-offs, and what the concrete next steps are."
        ),
        backstory=(
            "You run code review sessions for a small ML platform team. "
            "You are the one who decides what actually blocks a merge "
            "versus what goes in the backlog. You value a short, "
            "actionable report over an exhaustive one."
        ),
        llm=llm,
        verbose=True,
    )

    return critic, defender, arbiter


def build_tasks(
    critic: Agent,
    defender: Agent,
    arbiter: Agent,
    file_path: str,
    code: str,
) -> tuple[Task, Task, Task]:
    critic_task = Task(
        description=(
            f"Review the following Python file from the ChangeRisk "
            f"project, located at `{file_path}`.\n\n"
            f"```python\n{code}\n```\n\n"
            "List concrete problems only: bugs, code smells, risky "
            "assumptions, missing input validation, error handling "
            "gaps, or maintainability issues. For each one, reference "
            "the specific function or block it occurs in. Do not "
            "comment on style/formatting preferences (e.g. spacing) — "
            "focus on behavior and risk."
        ),
        expected_output=(
            "A numbered list of concrete issues, each with: the "
            "function/block affected, what the problem is, and why it "
            "matters."
        ),
        agent=critic,
    )

    defender_task = Task(
        description=(
            "You will receive the Critic's list of issues for the same "
            "file. For each issue, provide the most plausible design "
            "justification you can find in the code itself (naming "
            "conventions, comments, surrounding structure, the project's "
            "stated goals). If an issue has no reasonable justification, "
            "say so explicitly instead of inventing one."
        ),
        expected_output=(
            "A response to each of the Critic's numbered issues: either "
            "a justification, or an acknowledgment that the criticism "
            "is valid."
        ),
        agent=defender,
        context=[critic_task],
    )

    arbiter_task = Task(
        description=(
            "You will receive the Critic's issues and the Architect's "
            "responses for the same file. Produce a final markdown "
            "report with three sections:\n\n"
            "## Must Fix\n"
            "## Acceptable Trade-off\n"
            "## Notes\n\n"
            "Classify every issue from the discussion into exactly one "
            "of these sections, with a one-line justification for the "
            "classification. Keep the whole report under 400 words."
        ),
        expected_output=(
            "A markdown report with the three sections above, covering "
            "every issue raised by the Critic."
        ),
        agent=arbiter,
        context=[critic_task, defender_task],
    )

    return critic_task, defender_task, arbiter_task


def run_audit(file_path: str) -> str:
    """
    Runs the full Critic -> Defender -> Arbiter crew over a single
    file and returns the Arbiter's final markdown report as a string.
    """

    code = Path(file_path).read_text(encoding="utf-8")

    llm = _build_llm()
    critic, defender, arbiter = build_agents(llm)
    tasks = build_tasks(critic, defender, arbiter, file_path, code)

    crew = Crew(
        agents=[critic, defender, arbiter],
        tasks=list(tasks),
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
