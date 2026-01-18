Lecture 11 — DAWID, LLMs and Feature Detection (25 slides)

1. DAWID Frontend in Action
   - Full-slide screenshot of the DAWID web interface in use

2. AI Centre Client–Server Architecture
   - Overall system architecture: frontend, backend, LLMs, tools, data

3. DAWID Frontend: Design Philosophy
   - Lightweight web interface
   - No SPA, no heavy frameworks
   - Clear separation from backend logic

4. Frontend Technology Stack
   - HTML for structure
   - CSS for layout and styling
   - JavaScript for interaction
   - PHP as gateway layer

5. File Upload Workflow
   - Automatic upload on file selection
   - JavaScript Fetch API
   - PHP upload handler
   - Backend file storage

6. Streaming Responses in the Browser
   - Fetch API with ReadableStream
   - Incremental token display
   - Live Markdown rendering
   - Improved user experience

7. Session and Interaction History (Frontend)
   - Chat-style message blocks
   - Browser-side session continuity
   - Visual persistence of interaction

8. DAWID Backend: Central Uvicorn Server
   - FastAPI application
   - Multi-worker Uvicorn deployment
   - Central entry point (dawid_server.py)

9. Backend Route Overview
   - LLM routes
   - Upload routes
   - Audio routes
   - Dataspace routes
   - API routes

10. LLM Streaming Routes
    - Unified streaming interface
    - Session history handling
    - Context assembly
    - Tool and function interception

11. Upload and File Management Routes
    - User- and session-specific folders
    - Editable files vs data sources
    - Integration with tools and RAG

12. Audio Routes: Speech-to-Text
    - Audio upload
    - Local or remote transcription
    - Text merged into standard LLM pipeline

13. User Management and Dataspaces
    - Private user folders
    - Shared group folders
    - Controlled access to knowledge bases

14. Retrieval-Augmented Generation (RAG)
    - Local FAISS indices
    - Topic-aware document selection
    - Context injection into prompts

15. Available LLM Models and Capability Tiers
    - FAST, CORE, PRO, ULTRA tiers
    - OpenAI, Claude, Gemini, LLaMA, Mistral, Mixtral, GPT-OSS
    - One best model per tier and supplier

16. DAWID Model Aliases and Routing
    - Internal tier aliases (e.g. openai-core, llama-fast)
    - Resolution to concrete vendor models
    - Backend selection based on model ownership

17. Function Calling: Motivation
    - Text generation is not action
    - LLM decides what to do
    - System decides how to do it

18. Classical Function Calling
    - JSON-based function calls
    - Parsing and validation
    - Model-agnostic but fragile

19. Native Function Calling
    - Typed tool schemas
    - Explicit tool selection
    - Safer and cleaner execution

20. Function Calling inside DAWID
    - Detection during streaming
    - Mid-response execution
    - Function results fed back into context

21. LangGraph: From Calls to Workflows
    - Explicit state objects
    - Nodes as Python functions
    - Deterministic control flow

22. AI-Based Feature Detection: Problem Statement
    - What are weather fronts
    - Why automatic detection is hard
    - Motivation for deep learning

23. Input Data and Preprocessing
    - Six surface-level meteorological fields
    - Channel-wise normalization
    - Regular lat–lon grid

24. Model Architecture and Training
    - U-Net for semantic segmentation
    - Multi-class output
    - Supervised training with MLflow tracking

25. Inference Results: Front Detection in Forecasts
    - Application to ICON forecasts
    - Temporal evolution of fronts
    - Visual interpretation and consistency
