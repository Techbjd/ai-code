---
name: update-summary
description: Use when the user asks to update the conversation summary, or after completing any significant task (training, testing, debugging, new feature). Triggers on keywords like "update summary", "update conversation", "log session", "update diary". Also use after running scripts/pretrain.py, scripts/train_all.py, scripts/compare_all.py, or pytest.
---

# Update Conversation Summary

After completing any significant task in the VEGFR2 project, update both `CONVERSATION_SUMMARY.md` and `diary.md` to maintain session continuity.

## When to Update

Update after:
- Running training scripts (`scripts/train_all.py`, `scripts/pretrain.py`, `scripts/compare_all.py`)
- Running tests (`pytest`)
- Fixing bugs or refactoring code
- Adding new features or models
- Any significant change to the project

## Steps

### 1. Read Current State

```bash
# Read current conversation summary
cat CONVERSATION_SUMMARY.md

# Read current diary
cat diary.md

# Check what files changed
git status --short
git diff --stat
```

### 2. Update CONVERSATION_SUMMARY.md

Update these sections as needed:

- **Section 1 (PROJECT EVOLUTION)**: Add new session entry with number, date, and description
- **Section 3 (ALL MODELS)**: Add new models if any
- **Section 8 (ALL FILES)**: Add new files
- **Section 9 (RESULTS HISTORY)**: Add new AUC/MCC results
- **Section 10 (BUGS FIXED)**: Add any new bugs fixed
- **Section 11 (TEST RESULTS)**: Update test counts
- **Section 14 (NEXT STEPS)**: Update status (TODO -> DONE)

### 3. Update diary.md

Add a new session entry with this format:

```markdown
## Session N: Title (YYYY-MM-DD)

### Goal
What was the objective of this session.

### Tasks Completed
- [x] Task 1
- [x] Task 2

### Results
Quantitative results (AUC, MCC, test counts, etc.)

### Key Decisions
Important design choices made.

### Problems Encountered
Issues faced and how they were resolved.

### Next Steps
What to do next.
```

### 4. Update Section Numbers

If adding new sections to CONVERSATION_SUMMARY.md, renumber all subsequent sections.

### 5. Verify

Ensure both files are consistent and all links/references are correct.
