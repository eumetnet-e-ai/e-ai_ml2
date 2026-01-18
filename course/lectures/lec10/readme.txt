Lecture 10 — Agents and Coding with LLMs
=======================================

Slide 01 — From Scripts to Agents: Why AI Agents Exist
- Limits of classical scripts and pipelines
- Why LLMs change automation
- LLM vs tool vs agent
- Human-in-the-loop paradigm

Slide 02 — What Is an AI Agent? Core Concepts and Mental Models
- Perception, reasoning, action, memory
- Control loops and stopping criteria
- Stateless vs stateful systems
- Typical failure modes of agents

Slide 03 — LLM Capabilities and Limits in Practice
- What LLMs are good at
- What LLMs systematically fail at
- Hallucinations and overconfidence
- Determinism, temperature, reproducibility

Slide 04 — Prompting for Code Generation
- Prompt constraints for code-only output
- System vs user prompts
- Structuring requirements
- Common failure patterns

Slide 05 — Manual Coding with LLMs (Human in the Loop)
- Inspect–edit–execute workflow
- Trust boundaries
- When humans must intervene
- Productive collaboration patterns

Slide 06 — Manual Coding with LLMs: Example UI Interaction
- Typical web-based LLM interfaces
- Prompt → code → inspection cycle
- Why this is the baseline workflow

Slide 07 — Executing Generated Code Safely
- exec vs subprocess
- File-based execution
- Sandboxing concepts
- Security risks and mitigations

Slide 08 — Error Handling and Feedback Loops
- Capturing tracebacks
- Feeding errors back to LLMs
- Retry limits
- Detecting dead-end loops

Slide 09 — The First Coding Agent: Self-Correcting Loops
- Minimal autonomous coding agent
- Success criteria
- Logging and reproducibility
- Why this already qualifies as an agent

Slide 10 — From Prompts to Programs: Abstraction Boundaries
- Separating intent from execution
- Why structure matters
- Prompt brittleness
- Reusable design patterns

Slide 11 — Why Agent Frameworks Exist
- Scaling beyond ad-hoc scripts
- Reusability and structure
- Observability and control
- What frameworks do *not* solve

Slide 12 — Survey of Agent Frameworks (Critical Overview)
- Open-source vs commercial
- Maturity and stability assessment
- Research vs production readiness
- When not to use frameworks

Slide 13 — LangChain: Motivation and Architecture
- What LangChain solves
- Chains, prompts, tools, memory
- LLM abstraction layer
- Strengths and limitations

Slide 14 — Prompt Templates and Tool Integration with LangChain
- PromptTemplate design
- Tool calling concepts
- Binding Python functions
- Debugging LangChain chains

Slide 15 — Memory and Context Management
- Short-term vs long-term memory
- Chat history vs explicit state
- Context window limitations
- Memory failure modes

Slide 16 — Why Control Flow Matters for Agents
- Linear chains vs branching logic
- Conditional execution
- Retry paths
- Observability and inspection

Slide 17 — LangGraph: Graph-Based Agent Design
- Motivation for LangGraph
- Nodes, edges, and state
- Comparison to LangChain chains
- Design philosophy

Slide 18 — Explicit State Design for Agents
- Typed state schemas
- Contracts between nodes
- Avoiding hidden state
- Testing state transitions

Slide 19 — Designing Agent Nodes
- Pure functions vs LLM calls
- Tool nodes
- Side effects and isolation
- Unit testing nodes

Slide 20 — Task Flow and Graph Execution
- Entry and exit points
- Sequential and conditional flows
- Error propagation
- Debugging graph execution

Slide 21 — Building a LangGraph-Based Coding Agent
- Replacing loops with graphs
- Controlled retries
- Structured failure handling
- Loop-based vs graph-based agents

Slide 22 — LangGraph Weather Forecast Assistant (Overview)
- Parsing natural-language queries
- Data access as tools
- Visualization nodes
- End-to-end execution flow

Slide 23 — LangGraph Weather Forecast Assistant (Flow Detail)
- State evolution
- Node responsibilities
- Failure handling
- Output generation

Slide 24 — Multi-Agent Systems: When One Agent Is Not Enough
- Motivation for multiple agents
- Coordination vs independence
- Communication patterns
- Failure amplification

Slide 25 — CrewAI: Role-Based Agent Collaboration
- Roles and responsibilities
- Task delegation
- Strengths and weaknesses
- Practical limitations

Slide 26 — Agent Evaluation and Reliability
- Measuring success
- Regression testing agents
- Versioning prompts and graphs
- Monitoring and logging

Slide 27 — Tasks, Persistence, and Responsibility Management
- Task databases
- Ownership of results
- Reproducibility and replay
- Auditing agent decisions

Slide 28 — Operating Agents Safely
- Security boundaries
- Resource limits
- Human override mechanisms
- Compliance considerations

Slide 29 — From Prototypes to Systems
- Scaling considerations
- Maintenance costs
- Organizational impact
- Long-term robustness

Slide 30 — Summary and Outlook
- What agents can and cannot do
- Design principles to remember
- Open research questions
- Where this fits in larger systems
