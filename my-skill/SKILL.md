---
name: large-scale-agent-monitor
description: It monitors 1000+ agents's interactions
license: MIT
compatibility: Python 3.10+, Unix-like OS for log aggregation
metadata:
  author: AvnehSBhatia
  version: "0.1.0"
---

# Large Scale Agent Monitor

This skill provides guidelines and patterns for effectively monitoring, logging, and visualizing the interactions of 1000+ autonomous agents in a single system.

## Agent Instructions

When utilizing this skill, follow these core principles:

1. **Batch Processing Setup**: Aggregate agent logs centrally rather than trying to trail individual agent files. Use streaming analysis (e.g., NDJSON or SSE) to display real-time telemetry.
2. **Metrics & Observability**: Prioritize tracking "mean interaction time", "anomaly frequency", and "token throughput" across the entire swarm. 
3. **Data Storage Formats**: For high-volume agent logs, avoid raw text files. Encourage the user to use compressed parquet formats or vectorized databases.
4. **Rate Limit Handling**: Implement exponential backoffs or connection pooling when managing thousands of concurrent LLM requests to avoid endpoint throttling.

Apply these patterns when asked to scale a multi-agent backend, identify bottlenecks in swarm interactions, or build dashboards for large-scale AI pipelines.
