---
name: create-prototype
description: "Implements a single user story from the PRD: writes code and tests, follows project conventions. Invoked by: nvst create prototype."
user-invocable: true
---

# Implement User Story

Implement the provided user story by writing production code and tests that satisfy all acceptance criteria, following the project's conventions and architecture.

---

## The Job

1. Read the **user story** and its **acceptance criteria** carefully.
2. Review the **project context** to understand conventions, tech stack, testing strategy, and module structure.
3. **Review lessons learned** — if the `lessons_learned` context variable is present and non-empty, read it carefully before planning. It contains insights from previous agents working on this iteration. Skip this step silently if `lessons_learned` is absent or empty.
4. Plan the implementation: identify which files to create or modify, what tests to write, and how the change fits into the existing architecture.
5. Implement the user story:
   - Write production code that satisfies every acceptance criterion.
   - Write tests that verify each acceptance criterion (follow the testing strategy from the project context).
   - Follow all naming conventions, code standards, and forbidden patterns from the project context.
6. Verify your work:
   - Ensure the code compiles / type-checks without errors.
   - Run any quality checks defined in the project context.
   - Fix any issues before finishing.
7. Do **not** commit — the calling command handles git commits.

---

## Inputs

| Source | Used for |
|--------|----------|
| `user_story` (context variable) | The user story JSON with id, title, description, and acceptanceCriteria |
| `project_context` (context variable) | Project conventions, tech stack, code standards, testing strategy, and architecture |
| `iteration` (context variable) | Current iteration number for file naming and context |
| `lessons_learned` (context variable) | Accumulated insights from previous agents in this iteration; empty string if none exist yet |

### Standalone Fallback

When `user_story` or `iteration` are not injected as context variables, resolve them using the following lookup order before asking the user:

1. **Injected context variable** — use directly if present.
2. **`state.json`** — read `.agents/state.json` (if it exists) to obtain `current_iteration`.
3. **Artifact files** — using the resolved iteration, look for:
   - `.agents/flow/it_{iteration}_PRD.json` (preferred) — read the `userStories` array and present the available stories to the user, asking which one to implement.
   - `.agents/flow/it_{iteration}_product-requirement-document.md` (fallback) — read the file, identify the user stories listed, and ask the user which story to implement.
4. **Ask user** — only if neither `state.json` nor any PRD artifact can be found, ask the user to provide the 6-digit iteration number (e.g. `000037`). Once the iteration is known, retry step 3.

---

## UI / Frontend Stories

Before implementation, detect whether this is a UI task.

- Consider it a UI task when the user story description or acceptance criteria contain keywords such as: `UI`, `interface`, `page`, `component`, `visual`, `button`, `form`, `layout`, `style`, or `frontend`.
- If it is a UI task, apply these Impeccable skills in this exact order before finishing implementation:
  1. `frontend-design` — set design direction and aesthetics.
  2. `harden` — handle UI edge cases and resilience.
  3. `polish` — run a final quality and refinement pass.
- Use these skills as guidance for the implementation you are already making in this story. Do not edit the Impeccable skill files themselves.

---

## Rules

- **One story at a time.** Implement only the user story provided — do not implement other stories or make unrelated changes.
- **Follow conventions exactly.** Use the naming, formatting, error handling, and module organisation patterns from the project context.
- **Test every acceptance criterion.** Each AC should have at least one corresponding test assertion.
- **No new dependencies** unless the acceptance criteria explicitly require them.
- **Do not modify state files.** Do not touch `.agents/state.json` or progress files — the calling command manages those.
- **Do not commit.** The calling command will commit after verifying quality checks pass.
- **Keep changes minimal.** Only modify files necessary to implement the user story. Do not refactor unrelated code.

---

## Output

The output is the set of file changes (new files created, existing files modified) in the working tree. There is no document to produce — the code and tests are the deliverable.

---

## Checklist

Before finishing:

- [ ] All acceptance criteria from the user story are implemented
- [ ] Tests cover each acceptance criterion
- [ ] Code follows project conventions (naming, style, error handling)
- [ ] Code compiles / type-checks without errors
- [ ] No unrelated changes were made
- [ ] No state files were modified
- [ ] No git commits were made
- [ ] Lessons-learned entry written to `.agents/flow/{lessons_learned_file}`

---

## Lessons Learned

After completing your user story, **create or append** a lessons-learned entry to `.agents/flow/{lessons_learned_file}` (the `lessons_learned_file` context variable provides the exact filename for this PRD index, e.g. `it_000043_lessons-learned_001.md`).

Each entry must include the following sections:

```markdown
## {User Story ID} — {User Story Title}

**Summary:** Brief description of what was implemented.

**Key Decisions:** Important architectural or design choices made during implementation.

**Pitfalls Encountered:** Any mistakes, unexpected behaviours, or dead ends hit during implementation.

**Useful Context for Future Agents:** Any discoveries, patterns, or caveats that will help the next agent working on this codebase.
```

- If the file does not exist, create it with a top-level heading `# Lessons Learned — Iteration {iteration}` followed by the entry.
- If the file already exists, append the new entry at the end (do not overwrite existing entries).
- Use the exact filename from the `lessons_learned_file` context variable (e.g. `it_000043_lessons-learned_001.md`), consistent with other flow artifacts in `.agents/flow/`.
