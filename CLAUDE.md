## Agents

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
