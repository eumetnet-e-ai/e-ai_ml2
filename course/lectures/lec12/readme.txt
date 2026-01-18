Chapter 12 — MLFlow: Managing and Monitoring Training
Slide Plan (25 slides)

=====================================================
Part I — Motivation and Core Concepts
=====================================================

Slide 1 — Why Experiment Tracking Is Necessary
- Repeated training runs
- Hyperparameters, code versions, datasets
- Chaos without structure
- Reproducibility as a requirement

Slide 2 — Typical Problems Without Tracking
- Lost best parameters
- Non-reproducible results
- Missing metadata
- Knowledge locked in scripts

Slide 3 — What Is MLFlow?
- Open-source ML lifecycle tool
- Tracking, Projects, Models, Recipes
- Focus of this chapter: Tracking + Models

Slide 4 — MLFlow in the MLOps Pipeline
- Research → experimentation
- Validation → selection
- Registry → deployment
- Feedback loop

Slide 5 — Mental Model: Passive Infrastructure
- Training code is active
- MLFlow stores metadata
- UI is a viewer, not a controller
- Separation of concerns

=====================================================
Part II — MLFlow Tracking: Experiments and Runs
=====================================================

Slide 6 — Core Objects: Experiment and Run
- Experiment = container
- Run = single execution
- Named, timestamped, reproducible

Slide 7 — What Can Be Logged
- Parameters
- Metrics (time / step dependent)
- System metrics
- Artifacts (files, models, plots)

Slide 8 — Tracking URI and Storage Targets
- Local filesystem
- Remote server
- URI defines destination
- Transparent to training code

Slide 9 — Minimal Logging Example
- set_experiment
- start_run
- log_param
- log_metric

Slide 10 — Inspecting Results in the UI
- Compare runs
- Plot metrics
- Filter by parameters
- Live and post-hoc analysis

=====================================================
Part III — Logging a Real Training Workflow
=====================================================

Slide 11 — Example Problem: Wind Chill Regression
- Synthetic dataset
- Two inputs, one target
- Supervised regression task

Slide 12 — Data Generation and Preprocessing
- Random temperature and wind speed
- Physical wind chill formula
- Tensor preparation

Slide 13 — Model Definition
- Feedforward neural network
- Hidden layers
- Regression output

Slide 14 — Training Loop with MLFlow Logging
- Parameter logging
- Loss logging per epoch
- Long-running training visibility

Slide 15 — Logging Artifacts and Models
- Loss curve plots
- Model serialization
- Input-output signature

=====================================================
Part IV — Running MLFlow as a Server
=====================================================

Slide 16 — Local UI Mode
- mlflow ui
- Reads from mlruns/
- Single-user exploration

Slide 17 — Server Mode for Collaboration
- mlflow server
- Network binding
- Multiple users

Slide 18 — Background and Persistent Execution
- &
- nohup
- screen
- Operational robustness

Slide 19 — Authentication and Credentials
- Basic auth support
- Server-side config
- Client credential files

Slide 20 — Storage Backends and Scalability
- Backend store vs artifact store
- SQLite, PostgreSQL
- Local vs object storage

=====================================================
Part V — Advanced Features and Model Management
=====================================================

Slide 21 — Model Registry Concept
- Central model store
- Versioned models
- Full traceability to runs

Slide 22 — Model Promotion Workflow
- Register model
- Staging
- Production
- Rollback

Slide 23 — Exporting and Importing Experiments
- Migration between servers
- Experiment portability
- Reproducibility across systems

Slide 24 — Model Serving and REST APIs
- Serve registered models
- REST endpoint
- Decoupling training and inference

Slide 25 — From Research to Operations
- Tracking → registry → deployment
- MLFlow as MLOps backbone
- Reproducibility, transparency, collaboration
