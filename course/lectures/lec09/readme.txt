Lecture 9 — Diffusion and Flexible Graph Networks

Slide 01 — Introduction and Scope
- From deterministic prediction to learning distributions
- Generative models and structured learning
- Diffusion models and graph networks in one framework

--------------------------------------------------
Learning to Sample a Distribution (Conceptual Part)
--------------------------------------------------

Slide 02 — Why Learn a Distribution?
- Single predictions are not enough
- Uncertainty, variability, ensembles
- Sampling instead of point estimation

Slide 03 — From Regression to Generative Modeling
- Predicting y versus sampling x ~ p(x)
- Deterministic networks vs stochastic models
- Role of noise as a modeling tool

Slide 04 — Sampling as an Iterative Refinement Process
- Coarse guesses refined step by step
- Analogy: filtering, relaxation, physics solvers
- Stability through small updates

Slide 05 — Conditioning and Constraints
- Sampling with side information
- Conditioning on classes, observations, or physics
- Relevance for weather and climate applications

-----------------------------
Diffusion Networks (5 slides)
-----------------------------

Slide 06 — Diffusion Models: Core Idea
- Forward process: data → noise
- Reverse process: noise → data
- Learning the reverse dynamics

Slide 07 — Forward Diffusion Process
- Markov chain with Gaussian noise
- Noise schedule and convergence to N(0, I)
- Synthetic training data generation

Slide 08 — Reverse Diffusion and Denoising
- Learning p_θ(x_{t−1} | x_t)
- Neural network as denoiser
- Deterministic vs stochastic reverse steps

Slide 09 — Training Objective and Loss
- Denoising as supervised learning
- MSE loss over diffusion steps
- Relation to score-based models

Slide 10 — Diffusion for Weather and Climate
- Ensemble generation
- Conditional downscaling
- Uncertainty-aware field generation

-----------------------------------------
Graph Networks: Flexibility and Structure
-----------------------------------------

Slide 11 — Why Graph Networks?
- Irregular grids and sparse observations
- Beyond fixed Cartesian meshes
- Unified view of grids, meshes, stations

Slide 12 — What Makes Graph Networks Flexible?
- Decoupling geometry from learning
- Shared weights, variable graph sizes
- Same model on different domains

Slide 13 — Message Passing as Local Physics
- Neighborhood interactions
- Local aggregation, global behavior
- Relation to discretized PDEs

Slide 14 — Applying GNNs to Different Settings
- 1D grids, 2D meshes, sensor networks
- Changing resolution and domain size
- Coordinate-based vs coordinate-free inputs

------------------------------------
Worked Examples and Model Exploration
------------------------------------

Slide 15 — Example: Function Reconstruction from Sparse Data
- Sine and cosine functions
- Sparse observations on a grid
- Learning interpolation with GNNs

Slide 16 — Naive GNN Architecture
- Node features: value, mask, coordinate
- K-nearest-neighbor graph
- Two-layer message passing

Slide 17 — Generalization Experiments
- Larger domains than seen in training
- Higher-frequency target functions
- Observed strengths and failures

Slide 18 — Coordinate-Free GNN Example
- Removing absolute position information
- Learning purely from graph structure
- Improved scale generalization

Slide 19 — Exploring Message Passing
- Self vs neighbor contributions
- Information propagation across hops
- Interpretation of learned behavior

Slide 20 — Exploring Graph Structures
- Neighborhood size K
- Graph connectivity and smoothness
- Impact on reconstruction quality

--------------------------------------
Frameworks: Lightning and PyG
--------------------------------------

Slide 21 — Why Use PyTorch Lightning?
- Clean separation of model and training
- Reproducibility and scalability
- Research-to-production workflow

Slide 22 — PyTorch Lightning GNN Example
- Masked loss on unobserved nodes
- Automatic logging and device handling
- Clean training loop

Slide 23 — Why PyTorch Geometric?
- Native graph data structures
- Optimized message passing layers
- Expressive GNN building blocks

Slide 24 — PyTorch Geometric Example
- Graph convolution layers
- Explicit edge handling
- Inference and visualization

Slide 25 — Summary and Outlook
- Learning distributions via diffusion
- Flexible learning on graphs
- Toward hybrid AI–physics systems
