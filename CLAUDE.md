## Completion Discipline

**Nothing is done until it's fully done.** This is the #1 priority governing all work.

### Finish everything

Before declaring any task complete, verify against the source of truth:
- **Milestone/task has a checklist (ROADMAP, issue, spec)?** Every item must be addressed — implemented, tested, and checked off. If an item can't be done, explain why explicitly. Never silently skip items.
- **Tests are listed?** Every test must be implemented and run. A test that wasn't run is the same as a test that doesn't exist.
- **Code was written?** It must build and run successfully. "It should work" is not verification.

**Anti-pattern to avoid:** Implementing 80% of a task, declaring it "complete," and leaving the remaining 20% as implied future work. If the task says "verify f32, f16, int8" — verify all three, not just the first two.

### Self-review after implementation

After completing any non-trivial implementation, proactively review your own work before presenting it to the user:

1. **Completeness check:** Re-read the original requirements. Did you miss anything? Compare deliverables against the spec line by line.
2. **Build & test:** Run the build. Run the tests. If something fails, fix it — don't report partial success.
3. **Code quality scan:** Look at the code you wrote for:
   - Obvious bugs (off-by-one, null/nil handling, resource leaks)
   - Style consistency with existing code in the project
   - Missing error handling at boundaries
   - Hardcoded values that should be configurable
4. **Update tracking artifacts:** If there's a ROADMAP, checklist, or issue tracker — update it with actual results. Check boxes, note findings, record deviations.
5. **Propose improvements:** If you notice something worth improving but outside current scope, mention it explicitly rather than silently ignoring it.

If a self-review would take more effort than the implementation itself, at minimum do steps 1 (completeness) and 2 (build & test).

### When something doesn't work as expected

Document it honestly. If a test reveals a limitation (e.g., a backend doesn't support a feature), that's a finding — record it in the appropriate place (ROADMAP, report, or both) with the actual behavior and why. Don't paper over failures or quietly downgrade expectations.

---

## Testing Rules

### Adversarial Mindset

When writing tests, think like an attacker, not a developer. The goal
of a test is to BREAK the code, not to confirm it works.

For every function under test, ask:
1. What happens with nil/null/empty input?
2. What happens at zero? At one? At INT_MAX/SIZE_MAX?
3. What happens with malformed/truncated/oversized data?
4. What happens if called twice? Called concurrently? Called after release?
5. What error paths exist and are they all exercised?

Never write only happy-path tests. Every test file must include at
least as many adversarial/negative tests as positive tests.

### Edge Case Checklist (ZOMBIES)

Before considering a test suite complete, verify coverage of:

**Z - Zero:**
- Empty collections, zero-length data, zero count, zero duration
- nil/null for every pointer parameter
- Empty string ""

**O - One:**
- Single element/byte/sample/character
- Single iteration of any loop

**M - Many:**
- Large inputs (1M+ elements where feasible without slowing tests)
- Enough iterations to expose accumulation bugs (autoreleasepool, memory)

**B - Boundary:**
- Off-by-one: N-1, N, N+1 for every boundary N
- Integer limits: INT_MIN, INT_MAX, UINT_MAX, SIZE_MAX
- Float specials: NaN, +Inf, -Inf, -0.0, FLT_EPSILON, denormalized
- Buffer boundaries: exactly fits, one byte short, one byte over
- Type boundaries: signed/unsigned crossover, 32/64 bit limits

**I - Interface:**
- Every documented precondition violated
- Every documented error code triggered
- Every optional parameter as nil
- Every enum value including invalid cast values

**E - Exception/Error:**
- Every NSError** path exercised
- Error codes verified (not just error != nil)
- Error recovery: can the object still be used after an error?

**S - Simple scenarios first:**
- Build complexity incrementally
- If the simple case fails, do not add complexity

### Speed Requirements

**Unit tests MUST complete in < 1 second each.** No exceptions.

To achieve this:
- No model loading in unit tests (pass pre-loaded model or test without)
- No file I/O in pure logic tests (use in-memory data)
- No sleep, polling, or arbitrary waits
- No GPU computation in pure logic tests
- Minimal object construction -- only what the test needs

**Integration tests SHOULD complete in < 30 seconds each.**
- Share model loading across tests (load once in main)
- Use shortest audio that exercises the code path
- Prefer synthetic data over real audio files

**Slow tests (> 30s) require explicit justification:**
- End-to-end accuracy validation
- Performance benchmarks
- Memory leak detection over many iterations
- Concurrency stress tests

### Test Organization

Each test file tests ONE area of functionality. No monolithic test files.

Name tests to describe behavior and expected outcome:
```
test_{what}_{condition}_{expected_result}
Example: test_encode_zero_frames_returns_nil_error
```

Separate tests by speed tier:
- Tier 1 (< 5s total): No model, no GPU. Run on every build.
- Tier 2 (< 30s each): Needs model. Run on every commit.
- Tier 3 (minutes): Full pipeline. Run nightly or on PR.

### Test Structure

Use Arrange-Act-Assert with minimal Arrange:
```objc
static void test_example(SomeObject *obj) {
    // ARRANGE: only what this test needs, nothing more
    NSData *input = [NSData dataWithBytes:data length:len];

    // ACT: one action
    NSError *error = nil;
    NSData *result = [obj process:input error:&error];

    // ASSERT: focused, specific assertions
    ASSERT_TRUE(name, result != nil, fmtErr(@"failed", error));
    ASSERT_EQ(name, [result length], expectedLength);
}
```

Each test function tests ONE behavior. Do not combine multiple
behaviors in one test.

### Anti-Patterns to Avoid

**Tautological test:** Never compute expected values using the same
code being tested.

**Line hitter:** Every test MUST have at least one meaningful assertion.
A test that calls code without checking the result is not a test.

**Over-mocking:** If more than 2 mocks/stubs are needed, you are
probably testing the wrong thing. Prefer fakes with real behavior.

**Inspector:** Do not access private ivars or use runtime introspection
in tests. Test through public API only.

**Flaky test patterns:**
- Floating point comparison with == (use epsilon)
- Depending on hash map iteration order
- Depending on timing (use deterministic waits)
- Depending on filesystem state from other tests
- Using unseeded random values (always seed and log)

**Happy path only:** If a test file has only positive tests, it is
incomplete. Add adversarial tests before declaring done.

### Floating Point Testing Rules

Never compare floats with ==. Always use epsilon:
```objc
BOOL closeEnough = fabs(actual - expected) < 1e-5f;
ASSERT_TRUE(name, closeEnough, @"float mismatch");
```

Always test with: NaN, +Inf, -Inf, -0.0, very small
(denormalized), very large values.

### MRC Testing Rules

Every test that allocates objects must release them on ALL paths
including early returns.

Wrap test loops in @autoreleasepool:
```objc
for (int i = 0; i < N; i++) {
    @autoreleasepool {
        // test body
    }
}
```

For leak detection tests: measure RSS before and after, assert
delta < threshold.

### Concurrency Testing Rules

Never test concurrency with sleep + check. Use:
- dispatch_semaphore_wait
- dispatch_group_wait
- Completion handlers

When testing thread safety, run the concurrent operation at least
100 times to increase probability of exposing races.

Run tests under Thread Sanitizer (-fsanitize=thread) in CI.

### Sanitizer Requirements

Tests MUST pass under all three sanitizers (separate configurations):
- Address Sanitizer: memory corruption, use-after-free, buffer overflow
- Thread Sanitizer: data races, lock inversions
- Undefined Behavior Sanitizer: signed overflow, null deref, misalignment

Do not suppress sanitizer findings without documenting why.

---

## Anti-Patterns to Avoid

This project uses Obj-C++, C++17, Metal/MPS, and manual memory management (no ARC). Every code change must avoid these patterns. When reviewing code — your own or an agent's — check against this list.

### Concurrency & Data Races

- **Unprotected shared state between GCD and C++ threads.** CTranslate2 has its own thread pool. If GCD blocks access the same `Whisper` instance or `StorageView` without serialization, data races occur. Use a serial dispatch queue or separate model replicas per concurrent task.
- **Metal completion handler races.** `addCompletedHandler:` runs on an arbitrary Metal thread. Never modify shared state from a completion handler without synchronization — dispatch to a known queue.
- **C++ mutex held across `dispatch_async`.** A block may execute on a different thread than the one that locked. Use GCD serial queues for synchronization instead of mixing C++ mutexes with GCD.
- **`StorageView` shared across threads.** CTranslate2's tensor container is not thread-safe. Never pass a `StorageView` to one call while another thread reads or modifies it.
- **Concurrent encoding on the same `MTLCommandBuffer`.** Encoding is not thread-safe per command buffer. Create separate command buffers per thread via `MTLCommandQueue` (which is thread-safe).

### Time-Based Logic

- **Sleep-based polling.** Never use `usleep`/`sleep`/`[NSThread sleep...]` loops to wait for GPU or async work. Use `dispatch_semaphore_wait`, `MTLEvent`/`MTLSharedEvent`, or `addCompletedHandler:`.
- **`waitUntilCompleted` stalls.** Avoid blocking CPU while GPU executes when you could pipeline work. While GPU runs segment N, CPU should prepare segment N+1.
- **Hardcoded timeouts.** Don't hardcode timeouts for model loading or inference. Make them configurable or use cancellation tokens.

### Hardcoded Constants

- **Magic numbers.** Never write raw `3000`, `80`, `128`, `1500`, `0.5`, `2.4` etc. in code. Define named constants (`kMWDefaultChunkFrames`, `kMWCompressionRatioThreshold`) with a comment citing the source (e.g., Whisper paper, model config).
- **Hardcoded absolute paths.** Use configurable variables (CMake cache vars, environment vars, function parameters). No `/Users/...` in committed code.
- **Unchecked platform assumptions.** Don't assume Metal is available — check `MTLCreateSystemDefaultDevice()` and provide a clear error. Don't assume Apple Silicon (unified memory) without checking.

### God Objects & File Size

- **Single class doing everything.** `MWTranscriber` must delegate to `MWFeatureExtractor`, `MWTokenizer`, `MWDecodeLoop`, etc. — not absorb their logic. No class should exceed ~800 lines.
- **No file over 1000 lines.** If a file grows past this, split by responsibility.
- **No "Utils" dumping ground.** Group utilities by domain: `MWAudioMath.h`, `MWCompressionUtils.h`.
- **Header accumulation.** Don't put all types in one header. Use separate headers per concern: `MWTypes.h`, `MWErrors.h`, `MWSegment.h`, with an umbrella `MetalWhisper.h`.

### C++ Specific

- **Raw pointer ownership ambiguity.** Use `std::unique_ptr` for owning, raw pointers only for non-owning observers. Document ownership at every API boundary.
- **RAII gaps in exception paths.** Every `alloc`/`new`/resource acquisition must be cleaned up even if an exception is thrown. In MRC code, a C++ exception skips `[obj release]` — use `@try/@finally` or C++ RAII wrappers around Obj-C objects.
- **Object slicing.** Never store polymorphic objects by value. Use pointers or references.
- **Dangling references to temporaries.** Always copy `[nsString UTF8String]` into `std::string` — the `const char*` dies when the autorelease pool drains.
- **`shared_ptr` cycles.** Use `weak_ptr` for back-references. Prefer `unique_ptr` with non-owning raw pointers.
- **Header bloat.** Keep C++ includes strictly in `.mm` files. Never include CTranslate2 headers in public `.h` — this would break Swift imports (M10).

### Objective-C (MRC) Specific

- **Missing release on early-return paths.** Every `alloc`/`copy` must have a matching `release` on ALL code paths including error returns. Audit early returns for leaks.
- **Autorelease accumulation in loops.** Wrap every iteration of long-running loops (decode loop, batch processing) in `@autoreleasepool {}`.  For 1-hour audio this prevents hundreds of MB of leaked temporaries.
- **`@autoreleasepool` in init or around autoreleased return values.** Don't wrap init in `@autoreleasepool` (`[self release]` on failure risks premature dealloc). Don't wrap methods returning autoreleased objects — the pool drains the return value before the caller gets it.
- **Obj-C objects in C++ containers without retain.** `std::vector<NSString*>` doesn't call `retain` on insert or `release` on removal. Use `NSMutableArray` for Obj-C collections, or write a RAII wrapper.
- **Missing `@autoreleasepool` on C++ threads.** When CTranslate2 callbacks create Obj-C objects, there's no pool on the C++ thread. Wrap in `@autoreleasepool {}`.
- **Category method collisions.** Always prefix category methods: `mw_methodName`. Or prefer standalone functions over categories.
- **Forgetting `[super dealloc]`.** Must be the LAST line in `dealloc`, after all cleanup.

### Metal/GPU Specific

- **Unnecessary CPU↔GPU copies on unified memory.** Apple Silicon shares physical memory. Create `StorageView` on `Device::MPS` when the GPU will consume it. Pass `to_cpu=false` when the result feeds another GPU operation. Only copy to CPU for token/text processing.
- **No triple buffering for streaming.** For continuous audio (M13), maintain 3 mel spectrogram buffers with a `dispatch_semaphore_t(3)` to overlap CPU prep and GPU execution.
- **Blocking main thread with GPU work.** Always submit and wait for GPU work on background queues. Only dispatch UI updates to main.

### C++/Obj-C++ Interop

- **C++ exceptions crossing Obj-C boundaries.** Apple frameworks are not exception-safe — a C++ exception through `NSRunLoop`, GCD, or any Apple frame causes undefined behavior (typically abort). **Every** public Obj-C method calling C++ code must wrap in `try/catch`. No exceptions.
- **C++ types in public headers.** Never expose `std::string`, `StorageView`, etc. in `.h` files. Swift can't import them and it forces all importers to compile as Obj-C++. Use pimpl — declare C++ ivars in `@implementation`, not `@interface`.
- **Mixing memory models.** Don't `retain`/`release` C++ objects or `new`/`delete` Obj-C objects. Keep each in its own world.

### General

- **Cargo cult porting.** Don't transliterate Python line-by-line. Understand the algorithm, then implement idiomatically in C++/Obj-C++. Use Python as a spec, not a template.
- **Copy-paste error handling.** Extract a helper for the repeated try/catch→NSError pattern rather than duplicating it in every method.
- **Premature optimization.** Profile before writing SIMD intrinsics or Metal compute shaders for operations that aren't bottlenecks. For Whisper, the bottleneck is attention layers (GPU), not mel extraction or tokenization (CPU).
- **Golden hammer.** Don't route everything through Metal. Tokenization, string processing, file I/O belong on CPU. GPU is for matrix ops, FFT, model inference.
- **Boat anchor.** Don't build abstractions for features that don't exist yet (e.g., "generic backend" for CUDA support). YAGNI — this targets macOS Apple Silicon only.

---

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


