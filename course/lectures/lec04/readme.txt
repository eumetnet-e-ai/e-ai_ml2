Chapter 4 — Basics of Artificial Intelligence and Machine Learning (AI/ML)

Slide 1  — AI and ML as a Problem-Solving Approach
- Data-driven learning
- Function approximation instead of explicit rules
- Universal approximators
- Relation to scientific modelling (incl. MMS context)

Slide 2  — AI and ML as a Set of Tools
- PyTorch, TensorFlow, scikit-learn
- LLMs as a service
- On-premise vs cloud AI
- Role of compute (CPU/GPU/TPU)

Slide 3  — AI and ML as a New Paradigm for Interactivity
- Code assistants
- AI in research workflows
- Human–AI collaboration
- Productivity gains

Slide 4  — Critical Evaluation I: Reliability and Limits
- Dependence on training data
- Extrapolation failures
- Benchmark vs real-world performance
- Need for domain validation

Slide 5  — Critical Evaluation II: Bias, Transparency, Responsibility
- Bias in data and models
- Explainability and trust
- Human oversight in high-stakes decisions
- AI as decision support, not authority

Slide 6  — Torch Tensors: The Core Data Structure
- Tensor vs NumPy array
- Device awareness (CPU/GPU)
- requires_grad=True
- Short code example

Slide 7  — Automatic Differentiation (Autograd)
- Dynamic computation graphs
- Backpropagation without manual gradients
- Why this enables learning

Slide 8  — Data Handling in PyTorch: Dataset and DataLoader
- Why DataLoader exists
- Batching, shuffling, parallel loading
- Conceptual overview

Slide 9  — Batches Explained: What Comes Out of the DataLoader
- Shape of batches
- Why mini-batches help optimization
- Relation to memory and convergence

Slide 10 — Defining a Simple Neural Network
- Linear layers
- Activation functions
- Forward pass
- Minimal model code

Slide 11 — What the Model Actually Represents
- Parameters = learnable weights
- Nonlinear mappings
- Interpretation as function approximation

Slide 12 — Loss Functions: Measuring Error
- What a loss is
- Example: Mean Squared Error (MSE)
- Connection to optimization objective

Slide 13 — The Adam Optimizer
- Adaptive learning rates
- First and second moments
- Why Adam is widely used

Slide 14 — End-to-End Example: Sine Function Approximation (Overview)
- Goal of the example
- Dataset
- Model
- Training loop (conceptual)

Slide 15 — Sine Example: Code Walkthrough I
- Dataset generation
- Tensor conversion
- DataLoader usage

Slide 16 — Sine Example: Code Walkthrough II
- Model definition
- Loss and optimizer
- Training loop structure

Slide 17 — Sine Example: Training Outcome
- Loss over epochs
- Final approximation plot
- What “learning” means here

Slide 18 — DataLoader Without Shuffling
- Ordered batches
- Demonstration with structured data
- Why this can be problematic

Slide 19 — DataLoader With Shuffling
- Randomized batches
- Effect on training
- Comparison to previous slide

Slide 20 — From Prediction to Understanding
- Accuracy vs insight
- Why gradients matter
- Sensitivity analysis

Slide 21 — Defining a Classifier (Simple Version)
- Binary classification setup
- Simple network architecture
- Classifier code snippet

Slide 22 — Improving the Classifier (Better Version)
- Deeper network
- Nonlinearity
- Motivation for architectural choices

Slide 23 — Applying the Classifier to Data
- Training loop
- Prediction on grid
- Concept of decision boundary

Slide 24 — Gradient Fields and Decision Boundaries
- Gradients w.r.t. input
- Visualizing sensitivity
- Interpretation of gradient magnitude

Slide 25 — Take-Home Messages (Chapter 4)
- AI = differentiable approximation + optimization
- Tensors and gradients are the core
- Data handling matters
- Domain knowledge remains essential
