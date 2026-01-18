Lecture 14 — CI/CD for AI/ML: From Local Discipline to Operational Reality
=========================================================================

---------------------------------------------------------------------------
PART I — Why CI/CD Is Essential for AI/ML (Conceptual Motivation)
---------------------------------------------------------------------------

Slide 01 — Why CI/CD Is Not Optional for AI/ML
- AI/ML systems evolve continuously
- Models, data, code, and configuration all change
- Manual processes do not scale beyond experiments
- CI/CD as prerequisite for operational AI/ML

Slide 02 — AI/ML Is Not Special — But Failures Are Harder to See
- AI/ML uses standard software engineering principles
- The difference: failures are often silent
- Drift, degradation, and data issues are invisible without structure
- CI/CD provides traceability and early detection

---------------------------------------------------------------------------
PART II — Local Discipline First: Git Hooks as a Learning Tool
---------------------------------------------------------------------------

Slide 03 — Why Start Locally?
- Fast feedback loops matter
- Developers must test and validate before pushing
- CI platforms are slower and more expensive
- Local discipline accelerates development

Slide 04 — Git Hooks: The Core Idea
- Scripts executed automatically by Git
- Triggered by specific Git events
- Enforce rules before code leaves the laptop
- First line of defense

Slide 05 — When Git Hooks Are Executed
- pre-commit
- pre-push
- commit-msg
- How Git integrates hooks into its workflow

Slide 06 — Example: pre-commit Hook in Practice
- Running tests automatically
- Blocking commits on failure
- Immediate developer feedback
- No CI server involved

Slide 07 — What Happens During `git commit`
- Git detects the hook
- Script execution
- Success → commit proceeds
- Failure → commit aborted
- Deterministic, reproducible behavior

Slide 08 — Strengths of Git Hooks
- Extremely fast
- No external infrastructure
- Works offline
- Encourages clean development habits

Slide 09 — Limitations of Plain Git Hooks
- Local-only
- Not version controlled by default
- Easy to bypass
- Not enforceable across teams

Slide 10 — The pre-commit Framework
- Version-controlled hooks
- Shared configuration
- Same checks for all developers
- Bridges local and global workflows

Slide 11 — pre-commit as a Contract
- Developers agree on checks
- CI platforms enforce the same rules
- No surprises when pushing
- Local == global expectations

Slide 12 — Why This Matters for AI/ML
- Formatting, linting, and tests are trivial but essential
- Data processing code must be consistent
- Model code must be reproducible
- Errors should be caught before experiments run

---------------------------------------------------------------------------
PART III — From Local CI to Platform CI (Transition)
---------------------------------------------------------------------------

Slide 13 — Why Local Checks Are Not Enough
- No guarantee everyone runs them
- No neutral execution environment
- No audit trail
- No enforcement

Slide 14 — CI Platforms Take Over
- GitHub Actions
- GitLab CI
- Jenkins
- Same ideas, different ecosystems

---------------------------------------------------------------------------
PART IV — What We Actually Need for Real AI/ML Operations
---------------------------------------------------------------------------

Slide 15 — What Our Course Has Covered So Far
- Code development
- Model training
- Experiments and notebooks
- Basic containers

Slide 16 — What Is Still Missing
- Systematic validation
- Automated testing
- Artifact tracking
- Reproducible deployment

Slide 17 — Artifacts Matter More Than Code
- Models are artifacts
- Data snapshots are artifacts
- Containers are artifacts
- CI/CD manages artifacts, not just source code

Slide 18 — Reproducibility Beyond Git
- Git tracks code
- CI/CD tracks environments
- Registries track models and containers
- Full-chain reproducibility

Slide 19 — Deployment Is Not an Afterthought
- Training ≠ deployment
- Inference environments differ
- CI/CD prepares deployable units
- Rollback must be possible

Slide 20 — Why AI/ML Needs Structured Pipelines
- Experiments evolve into services
- Manual steps introduce risk
- Automation reduces cognitive load
- CI/CD is operational memory

Slide 21 — CI/CD as Risk Management
- Catch errors early
- Reduce blast radius
- Enable rollback
- Make failures observable

Slide 22 — HPC, GPUs, and Reality
- Specialized hardware
- Long runtimes
- CI/CD must adapt to constraints
- Separation of build and run

Slide 23 — ICON, AICON, Anemoi: What They Do Right
- pre-commit locally
- CI enforcement on platforms
- Integration tests
- Controlled environments

Slide 24 — Typical Failure Without CI/CD
- “Works on my machine”
- Undocumented environment changes
- Silent numerical errors
- Unreproducible results

Slide 25 — Takeaway: CI/CD Is Not Overhead
- CI/CD does not slow you down
- It enables speed at scale
- AI/ML without CI/CD remains experimental
- Operational AI/ML requires discipline
