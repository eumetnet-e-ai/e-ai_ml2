Lecture 15 — Anemoi: AI-Based Weather Modeling
=============================================

PART A — Motivation and Big Picture
-----------------------------------

Slide 01 — Why Anemoi?
- From NWP to ML-based Earth system models
- Why classical ML tooling is insufficient for weather/climate
- Positioning Anemoi in the ECMWF ecosystem

Slide 02 — Design Philosophy of Anemoi
- Declarative configuration
- Reproducibility and scalability
- Separation of concerns: data, graphs, models, training


PART B — Core Ingredients (Conceptual Building Blocks)
-------------------------------------------------------

YAML / Hydra / OmegaConf (3 slides)
----------------------------------

Slide 03 — YAML as the Configuration Backbone
- Why YAML for ML systems
- Structure, hierarchy, readability
- Static vs dynamic configuration

Slide 04 — Hydra: Composing Experiments
- Defaults, overrides, config groups
- Runtime switching of components
- Reproducible experiment management

Slide 05 — OmegaConf: The Runtime Config Object
- DictConfig and interpolation
- Validation and resolution
- Why OmegaConf sits underneath Hydra


Zarr and Scientific Data (3 slides)
-----------------------------------

Slide 06 — Scientific Data at Scale
- ERA, reanalyses, forecasts
- Why NetCDF alone is not enough
- Access patterns in ML training

Slide 07 — Zarr: Chunked, Parallel, Cloud-Ready
- Chunking, compression, metadata
- File-system vs object storage
- Why Zarr fits Anemoi

Slide 08 — From NetCDF/GRIB to Zarr
- Conversion workflows
- Variable conventions (e.g. `data`)
- Metadata and statistics


Graphs and Geometry (3 slides)
------------------------------

Slide 09 — Why Graphs in Weather ML?
- Non-Cartesian grids
- Spatial locality and interactions
- From grids to graphs

Slide 10 — Icosahedral and Geodesic Graphs
- ICON analogy
- TriNodes and refinement levels
- Resolution vs cost

Slide 11 — Edges, Neighbours, Topology
- kNN edges
- Edge index representation
- Reuse across experiments


PART C — Anemoi Packages (What Lives Where)
-------------------------------------------

Slide 12 — Anemoi Core: The Monorepo
- anemoi-core structure
- Why consolidation matters
- Long-term stability vs rapid evolution (⚠ warning slide)

Slide 13 — anemoi-training
- Training loop responsibilities
- PyTorch Lightning integration
- Strategies, callbacks, logging

Slide 14 — anemoi-models
- Model zoo and interfaces
- Layers vs architectures
- LightningModule conventions

Slide 15 — anemoi-datasets
- Dataset definitions and metadata
- Declarative dataset creation
- Validation and inspection tools

Slide 16 — anemoi-graphs
- Graph creation pipeline
- Node and edge builders
- Serialization and reuse


PART D — Training Pipeline (End-to-End View)
--------------------------------------------

Slide 17 — The Anemoi Training Pipeline
- From YAML to running experiment
- Who instantiates what, when
- High-level flow diagram

Slide 18 — Hydra Entry Point and CLI
- `anemoi-training train`
- Config search paths
- Validation before execution

Slide 19 — AnemoiTrainer Internals
- Cached properties
- Model, datamodule, graph
- Separation of orchestration and logic

Slide 20 — Lightning Handles the Loop
- Trainer.fit()
- Devices, distributed strategies
- Logging and checkpoints


PART E — Practical Workflow and Demo Preview
--------------------------------------------

Slide 21 — Dataset Creation Workflow
- GRIB → Zarr via anemoi-datasets
- Statistics and metadata
- Inspect and verify

Slide 22 — Configuration Validation
- Schema validation
- Debugging configs early
- Why validation matters operationally

Slide 23 — First Training Run (Conceptual)
- What happens when training starts
- What is computed where
- What is stored and logged

Slide 24 — Versioning, Reproducibility, Drift
- Why configs are first-class artifacts
- Dataset + model + graph = experiment
- Relation to MLOps concepts

Slide 25 — Outlook: From Demo to Operations
- Scaling up datasets and models
- HPC and distributed training
- How Anemoi fits into future NWP systems
