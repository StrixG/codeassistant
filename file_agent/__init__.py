"""File agent (Day 34): a goal-driven agent that reads, searches, and edits
files in the target repository via DeepSeek function calling.

Reuses the existing DeepSeek client, the Chroma code index (RagSearcher) and
the Tool/Registry/Executor plumbing from ``assistant.core`` — this package
only adds the file-mutating tool surface and the agent loop on top.
"""
