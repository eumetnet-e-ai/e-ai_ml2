Lecture 13 — MLOps: From Machine Learning to Operations

1. Title & Motivation
   - Why MLOps matters now
   - From ML experiments to reliable systems

2. Why ML Needs Operations
   - Models ≠ code
   - Data-driven behavior
   - Continuous change instead of static releases

3. What Is MLOps?
   - Extension of DevOps
   - Managing data, models, pipelines
   - Lifecycle perspective

4. DevOps vs MLOps
   - Software artifacts vs learned models
   - Determinism vs statistics
   - Releases vs retraining

5. Core Challenges in MLOps
   - Data versioning
   - Reproducibility
   - Drift and degradation
   - Monitoring beyond uptime

6. Typical MLOps Lifecycle
   - Data → features → training → validation
   - Registry → deployment → monitoring
   - Feedback loops

7. MLOps Pipeline (Conceptual View)
   - End-to-end flow
   - Automation points
   - Control gates

8. Roles in MLOps
   - Data scientists
   - ML engineers
   - DevOps / platform
   - Operations / users

9. Why MLOps Often Fails
   - Broken handovers
   - Hidden assumptions
   - Missing ownership

10. Monitoring ML Systems
    - Performance vs correctness
    - Drift detection
    - Retraining triggers

11. Reproducibility and Traceability
    - Data lineage
    - Model versions
    - Experiment tracking

12. Why Environments Matter
    - Dependency explosion
    - GPU / CPU differences
    - Library volatility

13. Containers: Core Idea
    - What a container is
    - Why VMs are insufficient
    - Reproducibility by construction

14. Docker for ML Systems
    - Image-based environments
    - Training vs inference containers
    - Local and cloud usage

15. Singularity / Apptainer for HPC
    - Security model
    - User-level execution
    - Why HPC prefers it

16. ML Inference in Containers
    - Model as a service
    - Batch vs REST
    - Decoupling training and ops

17. Minimal Inference Architecture
    - Model → container → interface
    - Conceptual only (no code yet)

18. Cloud Architectures for MLOps
    - On-prem, cloud, hybrid
    - Managed vs self-operated

19. Kubernetes and GitOps (Conceptual)
    - Declarative deployment
    - Versioned infrastructure
    - Rollbacks as first-class concept

20. Why Hybrid Architectures Dominate
    - HPC for physics & training
    - Cloud for services & AI
    - Clear interface boundaries

21. Weather Services: Special Constraints
    - Mission-critical operation
    - Fixed schedules
    - Public responsibility

22. NWP as Proto-MLOps
    - Automation before the term existed
    - Reproducibility by process
    - Conservative promotion to ops

23. DWD Operational Chain (ICON)
    - Global → regional → local
    - Deterministic and ensemble
    - Where AI fits in

24. BACY and NUMEX as DevOps Systems
    - Experiment → parallel routine → ops
    - Human gates instead of CI bots

25. From NWP to AI Systems (AICON)
    - What changes with ML
    - What stays the same
    - Why MLOps is unavoidable
