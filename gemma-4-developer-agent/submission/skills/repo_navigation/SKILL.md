---
name: repo_navigation
description: Navigates repository codebases using symbol search, induced graph subgraphs, and targeted line inspection.
---

# Repository Navigation Skill

This skill provides operational procedures for locating source symbols and call hierarchies across complex repositories.

## Best Practices
1. **Identify Entry Points**: Extract classes, methods, or CLI entry points directly from the issue description or stack traces.
2. **Constrain Exploration**: Use `search_similar_code(query)` and `get_code_neighbors(symbol)` to bound inspection to the 1-hop or 2-hop neighborhood of suspect symbols.
3. **Avoid Broad Sweeps**: Do not read full directories or execute unindexed file traversals. Use `read_file` with precise line slices (`start_line`, `end_line`).
