Plan for Lecture 18 — AI Data Assimilation (lec18.tex)

Overall goal:
Introduce AI-based data assimilation conceptually, position AI-Var relative to classical DA, explain the variational loss, and motivate why AI-Var is fundamentally different from “learning analyses”.

Structure and slide flow:
Total target: ~10–12 slides (concise, conceptual, no code yet)

--------------------------------------------------
lec18_01.tex — Motivation: Why Data Assimilation?
--------------------------------------------------
• Role of data assimilation in NWP
• Forecast skill depends on initial conditions
• Observations + model = analysis
• Key message: DA is the *information bottleneck* of NWP

--------------------------------------------------
lec18_02.tex — Classical Data Assimilation Cycle
--------------------------------------------------
• 3D-Var / 4D-Var / EnKF overview
• Background → analysis → forecast → new background
• Show classical DA cycle figure (aivar3.png)
• Emphasize Bayesian interpretation

--------------------------------------------------
lec18_03.tex — Two AI Paths for Using Observations
--------------------------------------------------
• Path A: observations directly inside NN forecasts
• Path B: observations used to compute an analysis
• Conceptual separation:
  – forecasting vs state estimation
• Position AI-Var clearly in Path B

--------------------------------------------------
lec18_04.tex — Why AI in Data Assimilation?
--------------------------------------------------
• Computational cost of classical DA
• Adjoint complexity, scalability issues
• AI promise:
  – fast inference
  – flexible operators
  – differentiable end-to-end systems

--------------------------------------------------
lec18_05.tex — From Variational DA to AI-Var
--------------------------------------------------
• Reminder: variational DA minimizes a cost functional
• Introduce J(x) with background + observation terms
• Explain meaning of B, R, H
• No implementation details yet

--------------------------------------------------
lec18_06.tex — Key Idea of AI-Var
--------------------------------------------------
• Replace minimization algorithm by a neural network
• Network outputs x_a directly
• Training driven by the *same cost function*
• Show AI-Var position figure (aivar2.png)

--------------------------------------------------
lec18_07.tex — Conceptual AI-Var Architecture
--------------------------------------------------
• Inputs: x_b and y
• Network produces x̂
• Loss = variational cost
• No “true analysis” needed
• Show conceptual diagram (aivar1.png)

• Explicit loss:
  – background mismatch weighted by B⁻¹
  – observation mismatch weighted by R⁻¹
• H(x̂) inside the loss
• Stress: physics/statistics embedded in loss

• Learned:
  – mapping (x_b, y) → x_a
• Not learned:
  – reanalysis fields
  – hand-crafted solvers
• Key conceptual shift: optimization → inference

Lecture 18 — Coding Examples (Slides 08–16)
Plan (text-only, copyable)

========================================
Notebook 1: 1_Inversion_1D.py
Theme: Classical 3D-Var vs AI-Var in the simplest possible setting
========================================

Slide 08 — 1D DA Setup: Truth, Background, Observations
- Purpose: Introduce the toy problem
- Content:
  • 1D periodic state
  • Truth: modulated sine
  • Background: shifted, smoothed, biased
  • Sparse noisy point observations
- Key message:
  “Minimal, fully controlled DA setup”

Slide 09 — Classical 3D-Var Solution in 1D
- Purpose: Reference solution
- Content:
  • Gaussian background covariance B
  • Point-sampling observation operator H
  • Closed-form / explicit 3D-Var solution
  • State plot + increment plot
- Key message:
  “Analysis balances background and observations”

Slide 10 — AI-Var Learns the 3D-Var Increment
- Purpose: Introduce AI-Var on the same problem
- Content:
  • Neural network predicts the increment δx
  • Training by minimizing the 3D-Var cost J
  • No analysis targets used
  • Comparison: ML increment vs 3D-Var increment
- Key message:
  “AI-Var reproduces the variational solution”

========================================
Notebook 2: 2_AI-VAR_1d.py
Theme: Learning the DA mapping and generalization
========================================

Slide 11 — Training AI-Var on Many 1D Cases
- Purpose: Move beyond a single case
- Content:
  • Randomized truth functions
  • Random background errors
  • Random observation locations
  • Fixed B and R
- Key message:
  “The network learns a general DA operator”

Slide 12 — AI-Var Generalization: Multiple Test Cases
- Purpose: Show robustness
- Content:
  • Multiple unseen test cases
  • State comparison: truth, background, 3D-Var, AI-Var
  • Increment comparison
- Key message:
  “One trained network, many analyses”

Slide 13 — Statistical Comparison: AI-Var vs 3D-Var
- Purpose: Quantitative validation
- Content:
  • RMSE over a fixed test set
  • Multiple independent training runs
  • Histogram / distribution comparison
- Key message:
  “AI-Var matches 3D-Var statistics”

========================================
Notebook 3: 3_assimilation_2d.py
Theme: Scaling AI-Var to spatially structured problems
========================================

Slide 14 — 2D Atmospheric Toy Problem
- Purpose: Increase realism
- Content:
  • 2D horizontal–vertical field
  • Structured background error
  • Sparse column observations
- Key message:
  “This already resembles NWP”

Slide 15 — Background Covariance in 2D
- Purpose: Explain spatial information spreading
- Content:
  • Gaussian B with horizontal and vertical scales
  • Impulse response B e_ij
- Key message:
  “B spreads information physically in space”

Slide 16 — 2D AI-Var vs 3D-Var Analysis
- Purpose: Final comparison
- Content:
  • Truth, background, 3D-Var analysis, AI-Var analysis
  • Error fields
- Key message:
  “Same statistics, different machinery”

========================================
Overall narrative
========================================
Slides 08–10: AI-Var works in a minimal setting
Slides 11–13: AI-Var generalizes and is statistically consistent
Slides 14–16: AI-Var scales to spatially structured problems

Transition after Slide 16:
From demonstrations → discussion of performance, scalability, and operational relevance.

