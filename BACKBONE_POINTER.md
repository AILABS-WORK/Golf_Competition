# BACKBONE — AI Orchestration Knowledge Base Pointer

> Feed this file into any project's CLAUDE.md / AGENTS.md / GEMINI.md to give agents access to the full skill/tool/plugin routing intelligence.

**Location:** `<JARVIS_ROOT>/BACKBONE/`
**Last updated:** 2026-03-23

---

## What the BACKBONE Is

The BACKBONE is an 80+ document orchestration knowledge base that maps **any task** to the best-fit **skill, plugin, MCP tool, or CLI command** for executing it. It contains:

- **31 intake sheets** analyzing community repos (skills, tools, MCPs, design systems)
- **39 trigger routes** mapping task types → optimal skill/tool combinations
- **Tool routing tables** for 8 categories of tools (source code, UI, deployment, database, testing, docs, security, payments)
- **15+ installed plugins** with activation rules
- **9 MCP servers** (GitHub, Firebase, Playwright, Stitch, Context7, GitNexus, Jarvis Channel, Scrapling, Notion)
- **15 agent classes** with complete skill/tool/plugin assignments (see Agent Orchestration Architecture)
- **1,300+ searchable skills** routed by `find-skills` at runtime
- **Design pipeline** (5 tools: Impeccable context → ui-ux-pro-max tokens → Stitch MCP generation → Impeccable quality → shadcn CLI components)
- **Quality pipeline** (6 tools: skill-security-auditor → pr-review-expert → tech-debt-tracker → performance-profiler → ci-cd-pipeline-builder → dependency-auditor)
- **Agent Orchestration Architecture** — full hierarchy, command cascade, context flow, live visibility map

---

## How to Use the BACKBONE (Decision Order)

When you receive a task, follow this order:

```
1. CLASSIFY the task → What type is it? (implementation, review, debug, design, docs, etc.)
2. CHECK TRIGGER MAP → Read BACKBONE/00_master/MASTER_TRIGGER_MAP.md → find the matching route (1-39)
3. CHECK TOOL ROUTER → Read BACKBONE/11_cli_mcp_plugins/TASK_TO_TOOL_ROUTER.md → find CLI/MCP tools for that route
4. LOAD SKILL → If the trigger map names a skill, load it (don't bulk-load all skills)
5. EXECUTE → Use the tool chain prescribed by the trigger route
```

### Quick Decision Tree

```
Is there a plugin command? (e.g., /feature-dev, /code-review, /review-pr, /commit)
  → YES: Use the plugin command first

Does a skill provide domain knowledge?
  → YES: Load the relevant skill

Need external context or live data?
  → YES: Use MCP (Context7 for docs, GitHub for repos, Stitch for design, Playwright for testing)

Need deterministic execution?
  → YES: Use CLI (git, gh, prisma, npx shadcn, npm, docker)

None of the above?
  → Direct prompting with BACKBONE patterns as context
```

---

## File Map — Which File Answers Which Question

### "What skill/tool should I use for X?"
| Question | Read This File |
|----------|---------------|
| What skill handles this task type? | `BACKBONE/00_master/MASTER_TRIGGER_MAP.md` |
| What CLI/MCP tool supports this task? | `BACKBONE/11_cli_mcp_plugins/TASK_TO_TOOL_ROUTER.md` |
| What plugins are installed? | `BACKBONE/11_cli_mcp_plugins/PLUGIN_REGISTRY.md` |
| What skills exist in the ecosystem? | `BACKBONE/00_master/MASTER_SKILL_REGISTRY.md` |
| Where can I find more skills? | `BACKBONE/03_skill_libraries/DISCOVERY_SOURCES.md` |
| **Full agent → skill → tool mapping** | `BACKBONE/10_deliverables/JARVIS_AGENT_ORCHESTRATION_ARCHITECTURE.md` |
| **Visual architecture diagram** | `BACKBONE/10_deliverables/jarvis-orchestration-diagram.html` |
| **Skills.sh recommendation report** | `BACKBONE/10_deliverables/SKILLS_SH_RECOMMENDATION_REPORT.md` |

### "Tell me about a specific repo/tool"
| Question | Read This File |
|----------|---------------|
| What does repo X contain/do? | `BACKBONE/09_repo_intake_sheets/<number>_<repo>.md` |
| Index of all analyzed repos | `BACKBONE/00_master/REPO_DEEP_INTAKE_INDEX.md` |
| What MCPs are configured? | `BACKBONE/11_cli_mcp_plugins/INSTALLED_TOOL_REGISTRY.md` |
| MCP configuration patterns | `BACKBONE/11_cli_mcp_plugins/RECOMMENDED_MCP_JSON_PATTERNS.md` |

### "How should I approach this work?"
| Question | Read This File |
|----------|---------------|
| Overall orchestration rules | `BACKBONE/00_master/MASTER_ORCHESTRATOR.md` |
| Debugging methodology | `BACKBONE/01_core_behavior/DEBUGGING_METHODOLOGY.md` |
| Governance / quality gates | `BACKBONE/01_core_behavior/GOVERNANCE_RULES.md` |
| Code review process | `BACKBONE/02_execution_pipelines/CODE_REVIEW_SYSTEM.md` |
| Agent task delegation | `BACKBONE/00_master/AGENT_TASK_DELEGATION.md` |
| Research protocol | `BACKBONE/06_research/RESEARCH_PROTOCOL.md` |

---

## Key Trigger Routes (Most Used)

| # | Task Type | Primary Tool/Skill | Key Command |
|---|-----------|-------------------|-------------|
| 3 | Debugging | yes.md + /debug | Escalation: 2→switch, 3→audit, 4→repro, 5+→handoff |
| 4 | Full project pipeline | yk-dev-pipeline (7 skills) | spec → plan → implement → review → test → docs |
| 7 | Code review | yk-dev-pipeline/code-review | `/code-review` or `/review-pr` |
| 8 | Testing | yk-dev-pipeline/testing | `npm test`, agent-browser for E2E |
| 12 | Skill discovery | antigravity, buildwithclaude | Search catalogs before creating new |
| 28 | UI/UX design | Stitch → frontend-design → shadcn | Full AI design pipeline |
| 32 | Visual docs | visual-explainer (8 commands) | `/generate-web-diagram` |
| 33 | Skill security | skill-security-auditor | `python3 scripts/skill_security_auditor.py` |
| 38 | PR review | pr-review-expert + GitNexus | Blast radius + 30-item checklist |
| 39 | UI components | shadcn CLI + Context7 | `npx shadcn@latest add <component>` |

---

## Installed Design Pipeline (5 Tools)

```
┌──────────────────┐    ┌─────────────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ /teach-impeccable│ →  │ ui-ux-pro-max       │ →  │ Stitch MCP   │ →  │ Impeccable       │ →  │ shadcn CLI   │
│ (gather context: │    │ (design tokens,     │    │ (AI generate │    │ (/audit          │    │ (add real    │
│  audience, brand,│    │  161 rules, BM25,   │    │  screens,    │    │  /normalize      │    │  components, │
│  personality)    │    │  design system gen)  │    │  export HTML)│    │  /polish         │    │  blocks,     │
└──────────────────┘    └─────────────────────┘    └──────────────┘    │  /animate)       │    │  charts)     │
                                                                       └──────────────────┘    └──────────────┘
```

### When to use which:
- **Impeccable `/teach-impeccable`:** Design CONTEXT (gather audience, brand, jobs-to-be-done → `.impeccable.md`)
- **ui-ux-pro-max:** Design DECISIONS (what colors, fonts, spacing, style to use → `MASTER.md`)
- **Stitch MCP:** Screen GENERATION (text/image → visual mockup → HTML export)
- **Impeccable `/audit /normalize /polish`:** Quality EXECUTION (audit, fix, refine — 20 commands)
- **shadcn CLI + MCP:** Component IMPLEMENTATION (install real, accessible components — 7 MCP tools)

---

## Installed Quality Pipeline (6 Tools)

```
┌──────────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ skill-security-      │ →  │ dependency-      │ →  │ pr-review-       │
│ auditor (gate new    │    │ auditor (CVE     │    │ expert (blast    │
│ skills before install│    │ scan + license)  │    │ radius + 30 items│
└──────────────────────┘    └─────────────────┘    └──────────────────┘
                                    ↓
┌──────────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ performance-         │ ←  │ tech-debt-       │ ←  │ ci-cd-pipeline-  │
│ profiler (CPU/mem/   │    │ tracker (scan    │    │ builder (detect   │
│ bundle/load testing) │    │ → score → dash)  │    │ stack → gen YAML) │
└──────────────────────┘    └─────────────────┘    └──────────────────┘
```

---

## Installed Plugins (15)

| Plugin | What It Does | Invoke With |
|--------|-------------|-------------|
| **feature-dev** | Full feature workflow (explore → architect → implement → review) | `/feature-dev` |
| **code-review** | Multi-agent code review with confidence scoring | `/code-review` |
| **pr-review-toolkit** | 6-agent PR review (code, tests, types, errors, comments, simplification) | `/review-pr` |
| **commit-commands** | Streamlined git commit + push + PR | `/commit`, `/commit-push-pr` |
| **frontend-design** | Anti-AI-slop UI guidance + Stitch MCP integration | Auto-triggers on UI tasks |
| **hookify** | Create hooks from conversation mistake patterns | `/hookify` |
| **superpowers** | 14 agentic skills (TDD, debugging, verification, parallel dispatch) | Auto-activates per task type |
| **ui-ux-pro-max** | 161 design rules, BM25 search, design system generator | `/ui-ux-pro-max` |
| **visual-explainer** | Terminal output → styled HTML pages (diagrams, slides, dashboards) | `/generate-web-diagram` etc. |
| **understand-anything** | 5-agent codebase analysis → knowledge graph + dashboard | `/understand` |
| **security-guidance** | File edit security scanner (XSS, injection, secrets) | Always active (hook) |
| **agent-sdk-dev** | Claude Agent SDK scaffolding | `/new-sdk-app` |
| **plugin-dev** | Plugin authoring toolkit | `/create-plugin` |
| **skill-codex** | Delegate to OpenAI Codex CLI for cross-tool analysis | Activates on Codex requests |
| **explanatory/learning** | Educational output style hooks | Always active (session-start) |

---

## MCP Servers (6)

| MCP | Purpose | Key Tools |
|-----|---------|-----------|
| **GitHub** | Repo ops, PRs, issues, code search | `create_pull_request`, `search_code`, `get_file_contents` |
| **Firebase** | Auth, hosting, Firestore | `firebase_get_project`, `firebase_list_apps` |
| **Playwright** | Browser testing, screenshots | `browser_navigate`, `browser_snapshot`, `browser_click` |
| **Stitch** | AI UI generation (Gemini) | `generate_screen`, `enhance_prompt`, `download_html` |
| **shadcn** | Component search, examples, audit | `search_items_in_registries`, `view_items_in_registries`, `get_audit_checklist` |
| **Context7** | Library/framework documentation | `resolve-library-id`, `query-docs` |
| **GitNexus** | Codebase knowledge graph | `query`, `context`, `impact`, `detect_changes` |

---

## CLI Tools Available

| Tool | Purpose | Key Command |
|------|---------|-------------|
| **shadcn** | UI components | `npx shadcn@latest add <component>` |
| **prisma** | Database ORM | `npx prisma generate`, `npx prisma db push` |
| **agent-browser** | Browser automation | `agent-browser navigate <url>`, `agent-browser snapshot` |
| **gh** | GitHub CLI | `gh pr create`, `gh issue list` |
| **git** | Version control | Standard git commands |
| **npm/npx** | Package management | `npm run dev`, `npx turbo build` |

---

## 31 Analyzed Repos (Intake Sheets)

Full intake sheets with one-sentence definitions, extraction targets, and risk assessments:

| # | Repo | Type | Status |
|---|------|------|--------|
| 1 | anthropics/skills | Official skill format reference | Foundational |
| 2 | sstklen/ai-md | Instruction compression (6-phase) | Candidate Golden |
| 3 | sstklen/yes.md | Governance + anti-slack (6-layer) | Candidate Golden |
| 4 | tanweai/pua | Debugging pressure overlay (L0-L4) | Approved-as-overlay |
| 5 | ykondrat/yk-dev-pipeline-skills | 7-skill SDLC pipeline | Candidate Golden |
| 6 | sstklen/5x-cto | Role-split dev methodology | Adapted |
| 7 | glittercowboy/taches-cc-resources | 27+ commands, 12 /consider:* models | Daily-Use |
| 8 | sickn33/antigravity-awesome-skills | 1,273+ skill catalog | Search-only |
| 9 | davepoon/buildwithclaude | 396 implementations + 20K index | Library+Discovery |
| 10 | daymade/claude-code-skills | 43 curated skills + skill-creator | Marketplace |
| 11 | posit-dev/skills | 18 vendor-maintained skills | Reference |
| 12 | travisvn/awesome-claude-skills | Curated discovery list | Discovery |
| 13 | ComposioHQ/awesome-claude-plugins | Plugin index (8 categories) | Discovery |
| 14 | GuDaStudio/skills | Multi-model collaboration (Gemini/Codex) | Approved |
| 15 | sstklen/opus-relay | Remote execution bridge | Experimental |
| 16 | sstklen/infinite-gratitude | Parallel research (1-10 agents) | Research-only |
| 17 | Lum1104/Understand-Anything | Codebase knowledge graph + dashboard | Approved |
| 18 | ~~washin-claude-skills~~ | Deleted → replaced by washin-playbook | N/A |
| 19 | sstklen/clawapi | API key manager + smart router | Infrastructure |
| 20 | sstklen/clawapi/drclaw | Collaborative debugging KB | Experimental |
| 21 | sstklen/washin-playbook | 7-chapter, 112 battle-tested skills | Approved |
| 22 | hesreallyhim/awesome-claude-code | 200+ tools/skills, CSV index | Discovery |
| 23 | gemini-cli-extensions/stitch | AI UI generation MCP (Google) | Installed |
| 24 | nextlevelbuilder/ui-ux-pro-max-skill | 161 design rules, BM25, design system gen | Installed |
| 25 | obra/superpowers | 14 agentic skills, verification gates | Installed |
| 26 | thedotmack/claude-mem | Memory (SQLite+Chroma, progressive disclosure) | Reference |
| 27 | vercel-labs/agent-browser | AI-native browser CLI (Rust, ref-based) | Golden |
| 28 | mar-antaya/my-claude-skills | 6 engineering quality skills | Installed |
| 29 | shadcn-ui/ui | 50+ components, CLI, MCP (7 tools), blocks, charts | In Use |
| 30 | pbakaus/impeccable | 20 design execution commands, 7 reference docs, anti-AI-slop | Approved |

---

## How to Add This to Another Project

Add the following to your project's `CLAUDE.md`:

```markdown
# BACKBONE — AI Orchestration Knowledge Base

This project has access to the Jarvis BACKBONE at `<path-to-jarvis>/BACKBONE/`.

## How to Route Tasks

1. Classify the task type
2. Read `BACKBONE/00_master/MASTER_TRIGGER_MAP.md` for skill routing (39 routes)
3. Read `BACKBONE/11_cli_mcp_plugins/TASK_TO_TOOL_ROUTER.md` for tool routing
4. Load only the relevant skill/intake sheet — never bulk-load

## Key Files
- Trigger Map: `BACKBONE/00_master/MASTER_TRIGGER_MAP.md`
- Tool Router: `BACKBONE/11_cli_mcp_plugins/TASK_TO_TOOL_ROUTER.md`
- Plugin Registry: `BACKBONE/11_cli_mcp_plugins/PLUGIN_REGISTRY.md`
- Skill Registry: `BACKBONE/00_master/MASTER_SKILL_REGISTRY.md`
- Repo Index: `BACKBONE/00_master/REPO_DEEP_INTAKE_INDEX.md`
- Discovery: `BACKBONE/03_skill_libraries/DISCOVERY_SOURCES.md`
```

---

## Rules for Using the BACKBONE

1. **Don't bulk-load.** Read only the file relevant to your current task.
2. **Follow the decision order.** Plugin command → Skill → MCP → CLI → Direct prompting.
3. **Check intake sheets** before installing any new tool. They contain risk assessments.
4. **Run security audit** before installing skills from unknown sources (`skill-security-auditor`).
5. **Use the trigger map** as the primary router — it maps 39 task types to optimal tools.
6. **Trust installed tools** over catalog listings. "Normalized" ≠ "installed."
