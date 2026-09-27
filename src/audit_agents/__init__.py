"""
Multi-agent code audit module.

Uses CrewAI to run three cooperating LLM agents (Critic, Defender,
Arbiter) over a source file from the ChangeRisk project, producing
a structured markdown audit report.

This module is experimental and separate from the production
ML/MLOps pipeline. It requires the extra dependencies listed in
`requirements-agents.txt` and a running LLM backend (Ollama by
default; see `.env.example`).
"""
