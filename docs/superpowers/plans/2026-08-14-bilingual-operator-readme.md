# Bilingual Operator README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create synchronized Chinese and English root README files that help an operator install, configure, run, inspect, and troubleshoot the current Auto-Tune project.

**Architecture:** `README.md` is the default Chinese entry and `README_EN.md` is the English equivalent. Both use the same section order, commands, capability boundaries, Mermaid flow, safety notes, and documentation links.

**Tech Stack:** GitHub-flavored Markdown, Mermaid, Windows batch commands, Conda, Python 3.10.

## Global Constraints

- Do not include real API keys, local absolute paths, datasets, weights, logs, or training artifacts.
- Do not claim Linux, YOLO11/26, model-structure adjustment, or multi-user hosting is complete.
- Present directory selection as the recommended data-ingestion workflow.
- State the fresh verified evidence: `160 passed`, with two existing sklearn PCA warnings.
- Do not embed the roadmap page-render PNG files; use Mermaid instead.
- Keep Chinese and English commands and facts identical.

---

### Task 1: Create the Chinese Operator README

**Files:**
- Create: `README.md`
- Reference: `auto_tune/config.template.yaml`
- Reference: `CLAUDE.md`
- Reference: `docs/roadmap_20260814.md`

**Interfaces:**
- Produces the default GitHub landing page.
- Links to `README_EN.md`, `auto_tune/config.template.yaml`, and roadmap/implementation documents.

- [ ] **Step 1:** Add the language switch and operator-focused project summary.
- [ ] **Step 2:** Add current capabilities and explicit support boundaries.
- [ ] **Step 3:** Add a Mermaid workflow from directory selection through analysis, training, tuning, audit, and unified history.
- [ ] **Step 4:** Add verified Windows setup, configuration, startup, and test commands.
- [ ] **Step 5:** Add step-by-step operator workflows for dataset analysis, manual training, diagnosis, auto-tuning, audit, and history.
- [ ] **Step 6:** Add troubleshooting, project structure, roadmap, and GitHub safety exclusions.

Expected: a reader can run the application without reading internal developer documents.

---

### Task 2: Create the English Operator README

**Files:**
- Create: `README_EN.md`
- Consume: section and command contract from `README.md`

**Interfaces:**
- Produces the English GitHub landing page.
- Links back to `README.md`.

- [ ] **Step 1:** Translate every user-facing section while preserving technical identifiers.
- [ ] **Step 2:** Keep code blocks, paths, links, support boundaries, and test evidence identical.
- [ ] **Step 3:** Check that operator terms map consistently: manual training, auto-tuning, guardrails, audit record, unified experiment history.

Expected: Chinese and English readers receive the same operational instructions and project status.

---

### Task 3: Validate README Safety and Consistency

**Files:**
- Verify: `README.md`
- Verify: `README_EN.md`

- [ ] **Step 1:** Check Markdown fences and Mermaid block balance.
- [ ] **Step 2:** Resolve every relative Markdown link against the repository.
- [ ] **Step 3:** Compare command blocks and facts across both languages.
- [ ] **Step 4:** Scan for API-key patterns, local absolute paths, dataset names, and generated artifact references.
- [ ] **Step 5:** Run the documented full test command and record the result without changing the claimed count unless evidence differs.

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected: `160 passed`; only the two known sklearn PCA warnings may remain.

No local Git commit is created in this plan. The README files join the first GitHub upload only after validation and secret-scope review.
