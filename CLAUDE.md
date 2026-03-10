# Agents

Use agents even if user did not ask you to do it explicitly
Agents folder is .claude/agents/ (in current project folder)

IMPORTANT: Before starting any subtask you MUST select the best agent and apply it
Always use agents for: code writing, analysis, design, debugging, testing, documentation
DO NOT use the Task tool to spawn subprocess agents - always use in-session agent loading instead.
Load ONLY ONE agent at a time - NEVER read multiple agent files in a single step
Agent instructions are TEMPORARY and apply only to the specific subtask at hand
Do NOT carry over agent-specific patterns, checklists, or conventions to unrelated subtasks

Agent Discovery Methods:
1. Check agent descriptions: Use Glob to list all agents: `*.md`
2. Search by domain: Use Grep to find agents by keyword in descriptions
3. When uncertain: Prefer specialized agents over general ones (e.g., python-pro over full-stack-developer for Python tasks)

User request execution workflow MUST be like this:
1. Check available agents in agents folder
2. Parse the user request and list ALL implied subtasks
3. Map each subtask to the best agent
4. For multi-step tasks, report to user: "I identified these subtasks: [list] and will use these agents: [list]"

Agent Selection Priority (when multiple agents could apply):
1. Most specialized agent wins (e.g., postgres-pro over database-optimizer for PostgreSQL)
2. For overlapping domains, choose based on primary task focus:
   - Performance issues → performance-engineer
   - Security issues → security-reviewer
   - Code quality → code-reviewer
3. For truly hybrid tasks, split into separate subtasks with different agents
4. Document your selection reasoning in the report to user

Subtask execution workflow MUST be like this:
1. Read the appropriate agent markdown file from agents folder using the Read tool (always fresh re-read)
2. Apply agent instructions to the CURRENT SUBTASK ONLY, complete fully, verify quality
3. DISCARD agent instructions — do not carry over to next subtask
4. Proceed with next subtask; once all finished, compose reports into one full report

Subtask Report Format:
1. Agent used: [agent-name]
2. Task description: [what was done]
3. Key findings/results: [bulleted list]
4. Files modified: [list of changed files]
5. Next steps/recommendations: [if applicable]

Agent instructions MUST be used for: code writing, analysis, design, debugging, testing, documentation, infrastructure, architecture. Use Glob/Grep on agents folder to find the best match by domain.


# Web Research

For any internet search:

1. Read agent instructions: `.claude/agents/web-searcher.md`
2. **ALWAYS** use `./.claude/tools/web_search.sh "query"` (or `.claude/tools/web_search.bat` on Windows). **NEVER use the built-in WebSearch tool** — all searches must go through the custom tool
   - **Multiple queries: combine into one call** — `web_search.sh "query1" "query2" "query3" -s 10` (parallel, cross-query URL dedup)
   - **Scientific queries: add `--sci`** for CS, physics, math, engineering (arXiv + OpenAlex)
   - **Medical queries: add `--med`** for medicine, clinical trials, biomedical (PubMed + Europe PMC + OpenAlex)
   - **Tech queries: add `--tech`** for software dev, DevOps, IT, startups (Hacker News + Stack Overflow + Dev.to + GitHub)
4. Synthesize results into a report

**Note**: Always use forward slashes (`/`) in paths for agent tool run, even on Windows.
Dependencies handled automatically via uv.


# Memory System

Two-tier: **Knowledge** (`knowledge.md`) permanent, **Session** (`session.md`) temporary.

| Question | Use |
|----------|-----|
| Will this help in future sessions? | **Knowledge** |
| Current task only? | **Session** |
| Discovered a gotcha/pattern/config? | **Knowledge** |
| Tracking todos/progress/blockers? | **Session** |

## Knowledge

```bash
memory.sh add <category> "<content>" [--tags a,b,c]
```

| Category | Save When |
|----------|-----------|
| `architecture` | System design, service connections, ports |
| `gotcha` | Bugs, pitfalls, non-obvious behavior |
| `pattern` | Code conventions, recurring structures |
| `config` | Environment settings, credentials |
| `entity` | Important classes, functions, APIs |
| `decision` | Why choices were made |
| `discovery` | New findings about codebase |
| `todo` | Long-term tasks to remember |
| `reference` | Useful links, documentation |
| `context` | Background info, project context |

**Tags:** Cross-cutting concerns (e.g., `--tags redis,production,auth`). **Skip:** Trivial, easily grep-able, duplicates.

**After tasks:** State "**Memories saved:** [list]" or "**Memories saved:** None"

**Other:** `search "<query>"`, `list [--category CAT]`, `delete <id>`, `stats`

## Session

Tracks current task. Persists until cleared.

**Categories:** `plan`, `todo`, `progress`, `note`, `context`, `decision`, `blocker`. **Statuses:** `pending` → `in_progress` → `completed` | `blocked`.

```bash
memory.sh session add todo "Task" --status pending
memory.sh session show                    # View current
memory.sh session update <id> --status completed
memory.sh session delete <id>
memory.sh session clear                   # Current only
memory.sh session clear --all             # ALL sessions
```

## Checkpoints

Save after every significant step. One active checkpoint (delete previous first). Under 500 chars.

```bash
memory.sh session add context "CHECKPOINT: [task] | DONE: [steps] | CURRENT: [now] | NEXT: [remaining] | FILES: [key files] | DECISIONS: [choices] | BUILD/TEST: [commands]"
```

After compaction: run `memory.sh session show` immediately to restore state.

**Rules:** One checkpoint at a time. Always include DONE and NEXT. Don't skip — losing state costs more.

## Multi-Session

Multiple CLI instances work without conflicts. Resolution: `-S` flag > `MEMORY_SESSION` env > `.claude/current_session` file > `"default"`.

```bash
memory.sh session use feature-auth        # Switch session
memory.sh -S other session add todo "..." # One-off
memory.sh session sessions                # List all
```


