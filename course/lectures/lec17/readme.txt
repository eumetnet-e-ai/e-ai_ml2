Lecture 17 — AICON Walkthrough (Notebook-driven slides)



Slide 02 — AI Weather Models: From Big Tech to Operations
- Big-Tech models (GraphCast, Pangu, FourCastNet, GenCast)
- Focus on end-to-end AI forecasting
- Typically trained offline, limited coupling to NWP systems
- Excellent performance, limited operational integration

Slide 03 — Operational AI Systems: AIFS, BRIS, AICON
- AIFS (ECMWF): global operational AI forecast system
- BRIS (Met Norway): national operational extension
- AICON (DWD): ICON-native AI emulator
- Common framework: Anemoi
- Different data, grids, and system coupling

Slide 04 — AICON as an Emulator (Visual)
- Emulator output fields (example)
- ICON triangular grid / mesh
- Key message: learning on the native ICON mesh

Slide 05 — Environment Setup (Conceptual)
- Python + PyTorch + Anemoi stack
- Zarr, ecCodes, earthkit
- Why reproducibility matters

Slide 06 — Manual Setup (What we actually do)
- Install from requirements.txt
- Verify versions (pip freeze)
- ecCodes definitions as part of runtime

Slide 07 — ecCodes and ICON Semantics
- Why standard ecCodes is not enough
- ICON / DWD definition tables (EDZW)
- Consequence of missing definitions (silent errors)

Slide 08 — Notebook Initialization
- Imports and basic configuration
- Deterministic seeding
- Environment sanity checks

Slide 09 — Data Sources Used
- ICON-DREAM reanalysis
- Zarr-based access
- Separation of training data vs inference data

Slide 10 — Variable and Metadata Handling
- ICON variables and naming
- shortName / longName resolution
- Why metadata consistency matters for ML

Slide 11 — From ICON Grid to Graph
- ICON mesh → graph abstraction
- Nodes, edges, connectivity
- Why graphs instead of Cartesian grids

Slide 12 — Model Configuration
- Anemoi model components
- Encoder / processor / decoder
- What is fixed, what is configurable

Slide 13 — Training Logic (High-level)
- Loss definition
- Temporal stepping
- Emulator perspective (state → state)

Slide 14 — Inference Workflow
- Loading trained model
- Running inference on ICON states
- Producing GRIB-compatible output

Slide 15 — Output and Verification
- Emulated vs reference fields
- Visual inspection
- First-order consistency checks

Slide 16 — What AICON Emulates (and what not)
- Emulated: forecast step
- Not emulated: DA, physics, numerics
- Position in the ICON workflow

Slide 17 — Operational Embedding
- Where AICON sits in production
- Interaction with ICON ecosystem
- Relation to AIFS / BRIS

Slide 18 — Key Lessons from the Walkthrough
- Environment matters
- Metadata matters
- Grid-native ML matters

Slide 19 — Outlook
- Scaling
- Coupling with DA
- Towards hybrid ICON–AI systems
