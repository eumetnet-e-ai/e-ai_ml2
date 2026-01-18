Lecture 6 — Slides completed so far (1–19)

1. Motivation: From Tokens to Language Models
   - What an LLM does at a high level
   - Text → tokens → probabilities

2. From Text to Tokens (Example)
   - Example sentence
   - Tokenization into subwords
   - Token IDs

3. Tokenization Details
   - Why words are split (e.g. “chased” → “chas”, “ed”)
   - Subword vocabulary intuition

4. From Token IDs to Embedding Vectors
   - Vocabulary size V
   - Embedding matrix E ∈ ℝ^{V × d_model}
   - Token ID → vector

5. Why Embeddings Are Needed
   - No meaning in integers
   - Meaning lives in learned vectors

6. Positional Encoding
   - Why order is missing
   - Additive positional encoding
   - x_i = e_i + p_i

7. Self-Attention: Core Idea
   - Tokens attend to tokens
   - Context-dependent representations

8. Queries, Keys, and Values — Intuition
   - Q: what am I looking for?
   - K: what do I offer?
   - V: what information do I pass?

9. Queries, Keys, and Values — Equations
   - Q = X W^Q
   - K = X W^K
   - V = X W^V
   - Dimensions explained

10. Scaled Dot-Product Attention
    - QKᵀ / √d_k
    - Softmax over tokens
    - Attention weights

11. Softmax
    - Definition
    - Turning scores into probabilities
    - Used in attention and output

12. Multi-Head Attention
    - Multiple attention heads
    - Concatenation
    - Output projection W^O

13. What Attention Can and Cannot Do
    - Linear weighted averages
    - No feature-wise nonlinearity

14. Position-Wise Feedforward Network (FFN)
    - FFN(x) = σ(xW₁ + b₁)W₂ + b₂
    - Applied per token
    - Why FFN is necessary

15. Residual Connections and Layer Normalization
    - X + Z_att
    - LayerNorm
    - Stabilization and gradient flow

16. Output Projection to Vocabulary
    - Z ∈ ℝ^{n × d_model}
    - Z_logits = Z W_out + b_out

17. Softmax over Vocabulary
    - Z_pred = softmax(Z_logits)
    - Probability distribution per token

18. Cross-Entropy Loss for Language Modeling
    - Target token index y_t
    - One-hot interpretation
    - L = −∑ log p_t(y_t)

19. Autoregressive Text Generation
    - Next-token prediction
    - Training vs inference
    - Feeding output back as input

--- ORIGINAL LIST---

1. Motivation: Why Large Language Models?
   - From NLP tasks to general-purpose language intelligence
   - LLMs as universal sequence models

2. LLMs as Sequence-to-Sequence Machines
   - Input tokens → output tokens
   - Autoregressive vs. bidirectional setups

3. From RNNs to Transformers
   - Limitations of RNNs / LSTMs
   - Parallelism and long-range dependencies

4. Transformer Architecture Overview
   - Encoder, decoder, encoder–decoder
   - Where GPT, BERT, T5 fit

5. Tokenization and Vocabulary
   - Words vs. subwords
   - BPE, WordPiece, SentencePiece

6. Embedding Tokens into Vectors
   - Embedding matrix
   - Dimensionality and semantic space

7. Positional Encoding
   - Why position information is needed
   - Sinusoidal encoding intuition

8. Self-Attention: Core Idea
   - Tokens attending to tokens
   - Contextual representations

9. Query, Key, Value Projections
   - Linear projections from embeddings
   - Role of Q, K, V

10. Scaled Dot-Product Attention
    - Similarity scores
    - Softmax normalization
    - Scaling by √d_k

11. Attention Matrix Interpretation
    - n × n interaction structure
    - Probabilistic weighting of context

12. Multi-Head Attention
    - Parallel attention subspaces
    - Concatenation and projection

13. Transformer Block Structure
    - Attention + residual + norm
    - Feedforward network

14. Output Projection to Vocabulary
    - Logits over vocabulary
    - Softmax probabilities

15. Language Modeling Loss
    - Next-token prediction
    - Cross-entropy loss

16. Training LLMs
    - Gradient descent and Adam/AdamW
    - Scale of data and compute

17. Encoder vs. Decoder Models
    - BERT-style understanding
    - GPT-style generation

18. Minimal Transformer in PyTorch
    - Positional encoding
    - Attention blocks
    - End-to-end model

19. Training a Small LLM Example
    - Dataset
    - Tokenization
    - Training loop

20. Text Generation Mechanics
    - Autoregressive decoding
    - Greedy vs. sampling intuition

21. Scaling Up: What Changes for Real LLMs
    - Billions of parameters
    - Distributed training
    - Mixed precision

22. Compute Cost of LLM Training
    - GPU hours
    - Academic vs. industrial scale

23. Running LLMs Locally
    - Ollama, llama.cpp, Hugging Face
    - CPU vs. GPU inference

24. Interacting with LLMs via APIs
    - REST interface
    - Python examples
    - Streaming responses

25. Practical Takeaway
    - Training is hard, usage is easy
    - LLMs as modular components
    - Focus on integration, not rebuilding
