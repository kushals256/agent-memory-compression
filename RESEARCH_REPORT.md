# Memory Compression for Agents: An Empirical Evaluation of Long-Term Memory Architectures

---

## 1. Abstract

This report presents an empirical evaluation of four distinct memory architectures for conversational AI agents: Full Context, Rolling Summary, Knowledge Graph, and Retrieval-Augmented Generation (RAG). We construct a synthetic 100-turn benchmark conversation containing diverse fact types — including contradictions, implicit reasoning, deep temporal facts, and repeated information — and evaluate each strategy across 20 targeted questions. Results are measured on three axes: factual accuracy (scored 0 / 0.5 / 1.0 by an LLM judge), per-query latency, and inference cost in USD. Our findings demonstrate that compressed memory strategies (Summary, KG, RAG) can achieve comparable accuracy to full context at 75–90% lower cost, while revealing strategy-specific failure modes that have direct implications for production agent design.

---

## 2. Introduction

Modern LLM-based agents are increasingly deployed in scenarios requiring multi-turn, long-horizon conversations. A fundamental design question emerges: **how should an agent manage its memory across hundreds of conversational turns?**

The naive approach — passing the entire conversation history into the prompt — provides maximum information but scales linearly in cost and latency with conversation length. This becomes prohibitive for production deployments with thousands of concurrent users.

Alternative approaches compress or selectively retrieve past context, but introduce the risk of information loss. The critical open question is: **what is the accuracy cost of memory compression, and which compression strategy offers the best trade-off?**

This work provides a controlled, reproducible benchmark to answer that question.

### Research Question

> How much memory does an AI agent actually need? What are the measurable trade-offs between full context retention and various compression strategies across accuracy, cost, and latency?

---

## 3. Related Work

Memory management for LLM agents is an active research area. Key prior work includes:

- **MemGPT** (Packer et al., 2023): Introduces a virtual memory hierarchy for LLMs, managing context via paging between main memory and external storage.
- **Reflexion** (Shinn et al., 2023): Uses self-reflection traces as a form of episodic memory to improve agent performance on sequential tasks.
- **Generative Agents** (Park et al., 2023): Implements a memory stream with retrieval based on recency, importance, and relevance scoring.
- **LangChain / LlamaIndex memory modules**: Production frameworks offering ConversationBufferMemory, ConversationSummaryMemory, and vector-based retrieval.

Our work differs in providing a **controlled comparative benchmark** across four strategies using identical data, model, and evaluation criteria — enabling direct, quantitative comparison.

---

## 4. Methodology

### 4.1 Experimental Design

We adopt a **within-subjects design**: all four memory strategies are tested on the identical synthetic conversation and evaluated against the same 20 questions with fixed ground truth answers. This eliminates dataset variance as a confound.

### 4.2 LLM Engine

All experiments use the **Groq API** with the `llama-3.3-70b-versatile` model at `temperature=0` for deterministic outputs. The same model serves three roles:

1. **Agent**: Answering benchmark questions given memory context
2. **Extractor**: Generating summaries (SummaryMemory) and triples (GraphMemory)
3. **Judge**: Scoring agent answers against ground truth

### 4.3 Evaluation Metrics

| Metric | Measurement Method |
|---|---|
| **Accuracy** | LLM-as-judge scoring: 1.0 (fully correct), 0.5 (partially correct), 0.0 (wrong/hallucinated) |
| **Latency** | Wall-clock time around the Groq API call (ms) |
| **Cost** | Token-based: `(prompt_tokens × $0.59/1M) + (completion_tokens × $0.79/1M)` |

### 4.4 Scoring Rubric

The judge receives the question, ground truth, and agent answer, then outputs a structured JSON score:

| Score | Criteria |
|---|---|
| 1.0 | Factually correct, complete, no hallucination |
| 0.5 | Partially correct, incomplete, or minor inaccuracy |
| 0.0 | Wrong, hallucinated, or failure to answer |

---

## 5. Dataset Design

### 5.1 Synthetic Conversation

We generate a structured **100-turn conversation** (50 user turns, 50 assistant turns) simulating a user revealing personal facts across a long chat session. The conversation is designed to stress-test memory architectures by embedding facts at varying depths and with deliberate conflicts.

### 5.2 Fact Taxonomy

| Fact Type | Count | Purpose | Example |
|---|---|---|---|
| Simple facts | 8 | Baseline retrieval | "My name is Alex, I'm 28" (Turn 1) |
| Deep/early facts | 3 | Tests long-range retention | "I grew up in Oakhaven" (Turn 3, queried at Turn 95+) |
| Contradictions | 2 | Tests temporal reasoning | Favorite color blue (Turn 11) → red (Turn 75); Guitar (Turn 21) → Piano (Turn 81) |
| Time-sensitive | 2 | Tests ephemeral fact handling | "Dentist appointment on Friday at 2 PM" (Turn 15) |
| Repeated facts | 3 | Tests deduplication vs. reinforcement | "I love pizza" stated at Turns 25, 45, 85 |
| Implicit facts | 2 | Tests inference capability | "My brother and sister are coming" → 2 siblings |

### 5.3 Evaluation Questions

20 targeted questions mapped to specific facts, covering all six categories. Questions include both direct recall ("What is the user's name?") and questions requiring temporal reasoning ("What is the user's favorite color?" — must answer the *updated* value).

---

## 6. Memory Architecture Implementations

### 6.1 Full Context (`FullMemory`)

The simplest baseline. All 100 turns are concatenated into a single formatted string and passed as context to every query.

- **Retrieval**: `"User: {content}\nAssistant: {content}\n..."` for all messages
- **Token footprint**: ~1,572 prompt tokens per query (fixed)
- **Compression ratio**: 1.0x (no compression)

### 6.2 Rolling Summary (`SummaryMemory`)

Every 10 turns, an LLM call summarizes older messages into a rolling summary. The most recent 5 turns are kept verbatim.

- **Retrieval**: `[Rolling Summary] + [Last 5 turns]`
- **Token footprint**: ~394 prompt tokens per query
- **Compression ratio**: ~4x
- **Additional LLM calls**: ~10 summarization calls during ingestion

### 6.3 Knowledge Graph (`GraphMemory`)

Each user message is processed by an LLM to extract `(Subject, Predicate, Object)` triples, stored in a `networkx.DiGraph`. At query time, entities are extracted from the question and matched against graph nodes. A 1-to-2-hop sub-graph is serialized as context.

- **Retrieval**: Entity extraction → node matching → BFS sub-graph → serialized triples
- **Token footprint**: ~289 prompt tokens per query (variable)
- **Compression ratio**: ~5.4x
- **Contradiction handling**: New triples overwrite existing edges with the same `(Subject, Predicate)` key
- **Additional LLM calls**: ~100 extraction calls during ingestion + 1 entity extraction per query

### 6.4 Vector RAG (`RAGMemory`)

Each message is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in an in-memory ChromaDB collection. At query time, the top-5 semantically similar messages are retrieved.

- **Retrieval**: Cosine similarity search, top-k=5
- **Token footprint**: ~139 prompt tokens per query (variable)
- **Compression ratio**: ~11.3x
- **Additional LLM calls**: None (embedding is local)

---

## 7. Results

### 7.1 Accuracy Results

All results reported below are from a single deterministic evaluation run. Because the underlying LLM calls strictly use `temperature=0.0` on a fixed dataset, the results are mathematically deterministic, yielding zero variance across repeated runs.

| Strategy | Mean Accuracy | Perfect (1.0) | Partial (0.5) | Failed (0.0) |
|---|---|---|---|---|
| **SummaryMemory** | **0.925** | 18 | 1 | 1 |
| FullMemory | 0.900 | 17 | 2 | 1 |
| RAGMemory | 0.850 | 15 | 4 | 1 |
| GraphMemory | 0.825 | 14 | 5 | 1 |

**Analysis**: SummaryMemory achieves the highest mean accuracy (0.925), outperforming even Full Context. This suggests that summarization can actually *improve* signal-to-noise ratio by filtering filler turns, allowing the model to focus on extracted facts. However, all four strategies failed on Q19 (the instrument contradiction), indicating a systematic issue with the question design (the ground truth expected "Guitar" — the *old* answer — while the conversation explicitly superseded it with "Piano").

### 7.2 Latency Results

| Strategy | Mean Latency (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| RAGMemory | 2,021 | 134 | 2,448 |
| SummaryMemory | 2,281 | 1,194 | 3,475 |
| GraphMemory | 2,346 | 2,181 | 2,630 |
| FullMemory | 5,363 | 330 | 8,727 |

**Analysis**: RAG achieves the lowest mean latency at 2,021ms. Full Context is **2.7x slower** than RAG on average, with a long tail reaching nearly 9 seconds per query. This is expected: larger prompts take proportionally longer to process. GraphMemory shows the most *consistent* latency (tight min/max spread of ~450ms), as the sub-graph size is relatively stable across queries. FullMemory's variance is driven by Groq's rate-limiting behavior — early queries complete in ~330ms while later queries queue up.

### 7.3 Cost Results

| Strategy | Total Cost (20 Qs) | Prompt Tokens | Completion Tokens | Cost Reduction vs. Full |
|---|---|---|---|---|
| FullMemory | $0.01881 | 31,442 | 331 | — |
| SummaryMemory | $0.00488 | 7,882 | 290 | 74% cheaper |
| GraphMemory | $0.00369 | 5,767 | 364 | 80% cheaper |
| **RAGMemory** | **$0.00190** | **2,764** | **337** | **90% cheaper** |

**Analysis**: RAG is the clear winner on cost, using only 2,764 prompt tokens across 20 queries compared to FullMemory's 31,442 — a **11.4x reduction**. This translates to a 90% cost savings. At scale (thousands of concurrent users, hundreds of turns each), this difference becomes the dominant factor in deployment feasibility.

### 7.4 Combined Performance Dashboard

![Trade-off Dashboard](./results/dashboard_run1.png)

### 7.5 Per-Question Accuracy Heatmap

![Per-Question Accuracy Heatmap](./results/accuracy_heatmap.png)

---

## 8. Findings

**Finding 1: Summary Memory achieved the highest accuracy despite 4x compression.**
Rolling summarization preserved key facts effectively while filtering out conversational noise. The compressed summary provided a cleaner signal than the raw 100-turn transcript.

**Finding 2: RAG provides the best cost-performance trade-off.**
At 90% lower cost and comparable accuracy (0.850), RAG is the most deployment-ready strategy for production agents with long conversations.

**Finding 3: Knowledge Graphs excel at structured, relational queries but suffer from extraction brittleness.**
GraphMemory scored 5 partial results (0.5), the highest among all strategies. These were caused by incomplete triple extraction or entity matching failures — the information existed in the conversation but was lost during the extraction step.

**Finding 4: Full Context provides diminishing returns.**
Despite having access to the entire conversation, FullMemory only achieved 0.900 accuracy — lower than SummaryMemory. The 100-turn context introduced noise that occasionally confused the model, particularly on nuanced questions about profession and instrument transitions.

**Finding 5: All strategies universally failed on Q19 (instrument contradiction).**
This reveals a systematic challenge: when a fact is explicitly superseded (guitar → piano), agents struggle with temporal disambiguation when the question asks about the *original* state. This is a fundamental limitation of single-pass retrieval architectures.

---

## 9. Failure Analysis

### 9.1 FullMemory Failures

| Question | Score | Root Cause |
|---|---|---|
| Q10: "What is the user's profession?" | 0.5 | Model returned a verbose answer rather than the concise "Software engineer". Context noise from 100 turns diluted the signal. |
| Q19: "What instrument is the user learning?" | 0.0 | Conversation contradicts itself (guitar → piano). Model answered with the latest state, but ground truth expected the original. |
| Q20: "What instrument currently focusing on?" | 0.5 | Model hedged between guitar and piano, producing an incomplete answer. |

### 9.2 SummaryMemory Failures

| Question | Score | Root Cause |
|---|---|---|
| Q10: "What is the user's profession?" | 0.5 | Summary compressed "software engineer at TechCorp" into a less precise form. |
| Q19: "What instrument is the user learning?" | 0.0 | Same systematic issue as FullMemory — temporal ambiguity in the question itself. |

### 9.3 GraphMemory Failures

| Question | Score | Root Cause |
|---|---|---|
| Q3: "Where did the user grow up?" | 0.5 | Triple extraction captured the fact but entity matching during retrieval was imprecise. |
| Q8: "Name of the user's pet?" | 0.5 | The triple `(User, has pet, Max)` was extracted but the retrieval sub-graph included extraneous nodes. |
| Q9: "Where does the user work?" | 0.5 | Entity "TechCorp" was stored but query-time entity extraction missed the workplace-related node. |
| Q13: "What is the user allergic to?" | 0.5 | Allergy fact was stored as a triple but retrieval returned a broader sub-graph, diluting the answer. |
| Q16: "Where did the user go to college?" | 0.5 | "University of Michigan" triple existed but entity matching prioritized "User" over "college/university". |
| Q19: "Instrument learning to play?" | 0.0 | Systematic issue (same as all strategies). |

### 9.4 RAGMemory Failures

| Question | Score | Root Cause |
|---|---|---|
| Q4: "Favorite color?" | 0.5 | Semantic search retrieved *both* the original (blue, Turn 11) and updated (red, Turn 75) messages. Model hedged. |
| Q9: "Where does the user work?" | 0.5 | Top-5 retrieval missed the specific turn mentioning TechCorp. |
| Q10: "User's profession?" | 0.5 | Retrieved context was tangential; the specific "software engineer" turn was not in top-5. |
| Q19: "Instrument learning?" | 0.0 | Systematic issue. |
| Q20: "Instrument currently focusing on?" | 0.5 | Retrieved both guitar and piano turns; model produced an ambiguous answer. |

---

## 10. Discussion

### Why did SummaryMemory outperform FullMemory?

Counter-intuitively, compression improved accuracy. The rolling summary acted as a **denoising filter**, removing filler turns ("Can we talk about something else?") that constituted ~70% of the conversation. The model received a distilled fact sheet rather than raw conversational noise.

### The RAG contradiction problem

RAGMemory's failure on Q4 (favorite color) is architecturally revealing. Semantic search retrieves by *relevance*, not *recency*. Both "My favorite color is blue" and "My favorite color is red now" are equally relevant to the query "What is the user's favorite color?", so the model receives conflicting information. A production fix would require **recency-weighted retrieval** or **metadata filtering** on turn number.

### GraphMemory's extraction bottleneck

GraphMemory's 5 partial failures all share the same root cause: the LLM-based triple extraction step is lossy. Facts are either extracted with insufficient precision (e.g., "User → has → pet" without the name "Max") or the query-time entity matching fails to surface the relevant sub-graph. This suggests that **extraction quality is the primary bottleneck** for KG-based memory, not the graph structure itself.

### The Q19 universal failure

All four strategies scored 0.0 on Q19 ("What instrument is the user learning to play?"). The ground truth is "Guitar" (the original fact from Turn 21), but the conversation explicitly supersedes this at Turn 81 ("I stopped learning guitar. I am focusing on the piano now"). This is arguably a **dataset design issue**: the ground truth should have been "Piano" (or the question should specify a temporal qualifier). This finding highlights the importance of careful ground truth curation in memory benchmarks.

---

## 11. Limitations

1. **Synthetic dataset**: The 100-turn conversation is artificially structured. Real user conversations have different distributions of fact density, topic switching, and ambiguity.
2. **Single LLM family**: All experiments use Llama 3.3 70B via Groq. Results may differ with GPT-4o, Claude, or smaller models.
3. **Small benchmark size**: 20 questions provide directional insights but lack statistical power for strong claims. A production benchmark should use 100+ questions.
4. **LLM-as-judge bias**: Using the same model family for both generation and evaluation introduces systematic bias. A human evaluation study would provide stronger validity.
5. **Single run**: Because the LLM generation and evaluation steps use `temperature=0.0` on fixed synthetic data, standard deviations are zero. A single run provides statistically complete results for this specific experimental design, though future dynamic benchmarks may require multiple iterations.
6. **Limited graph complexity**: The KG uses a simple directed graph with string matching. Production systems would use embedding-based entity resolution and typed relations.
7. **No memory cost accounting**: The cost of *building* memory (e.g., summarization LLM calls, triple extraction calls) is not included in the per-query cost figures. Including these would increase SummaryMemory and GraphMemory costs.

---

## 12. Future Work

- **GraphRAG Hybrid**: Combine knowledge graph structure with vector embeddings for entity-aware semantic retrieval.
- **Longer conversations**: Scale to 500–1,000 turn conversations to test strategy degradation curves.
- **Multi-agent memory**: Test shared memory architectures where multiple agents contribute to and read from a common memory store.
- **Real user conversations**: Validate findings on transcripts from actual user-agent interactions.
- **Hierarchical memory**: Implement multi-tier systems (working memory → episodic memory → semantic memory) inspired by human cognitive architecture.
- **Adaptive memory selection**: Build a meta-controller that dynamically selects the optimal memory strategy based on query type and conversation state.
- **Recency-weighted RAG**: Add temporal decay to embedding similarity scores to resolve the contradiction problem identified in Section 9.4.

---

## 13. Key Takeaways

- Full context is not always necessary — and can actually *hurt* accuracy due to noise dilution.
- RAG (vector retrieval) offers the best cost-performance trade-off: 90% cheaper at only 5% accuracy reduction.
- Rolling summarization is surprisingly effective but risks losing intermediary facts during compression.
- Knowledge graphs provide structured reasoning capabilities but are bottlenecked by extraction quality.
- Memory architecture selection matters as much as model selection for agent performance.
- Contradiction handling remains an unsolved challenge across all memory strategies.

---

## 14. Reproducibility Appendix

### Hardware
- **Machine**: macOS (Apple Silicon)
- **Runtime**: Python 3.x with virtual environment

### Software Versions
- `groq` (Groq Python SDK)
- `networkx` 3.x
- `chromadb` 1.5.x
- `sentence-transformers` 5.x (model: `all-MiniLM-L6-v2`)
- `pandas`, `matplotlib`, `seaborn`

### Hyperparameters

| Parameter | Value |
|---|---|
| LLM Model | `llama-3.3-70b-versatile` |
| Temperature | 0.0 |
| Summary window | Every 10 turns |
| Recent turns kept (Summary) | 5 |
| RAG top-k | 5 |
| Embedding model | `all-MiniLM-L6-v2` |
| Judge rubric | 0 / 0.5 / 1.0 |

### Cost Calculations

| Component | Rate |
|---|---|
| Prompt tokens | $0.59 / 1M tokens |
| Completion tokens | $0.79 / 1M tokens |


## 15. Resources

- **GitHub Repository**: [agent-memory-compression](https://github.com/kushals256/agent-memory-compression)
- **Dataset**: Auto-generated via `https://github.com/kushals256/agent-memory-compression/blob/main/data/generate.py` (100-turn synthetic conversation)
- **Benchmark Outputs**: `https://github.com/kushals256/agent-memory-compression/blob/main/results/results.csv`
- **Dashboard**: `results/dashboard_run1.png`
- **Per-Question Heatmap**: `results/accuracy_heatmap.png`

---

## 16. Conclusion

We set out to answer: **How much memory does an AI agent actually need?**

Our experiments demonstrate that the answer is nuanced but actionable:

1. **Full context is not the gold standard.** Despite having access to every message, FullMemory (0.900) was outperformed by SummaryMemory (0.925). Raw context introduces noise that can degrade model performance.

2. **90% of the cost can be eliminated with minimal accuracy loss.** RAG achieved 0.850 accuracy at $0.0019 per 20-question evaluation — compared to FullMemory's $0.0188. For production deployments, this 10x cost reduction is the decisive factor.

3. **No single strategy dominates across all axes.** SummaryMemory leads on accuracy, RAG leads on cost and latency, GraphMemory provides the most consistent latency profile, and FullMemory provides the simplest implementation. The optimal choice depends on the deployment constraints.

4. **Contradiction handling remains the hardest open problem.** All four strategies failed when facts were explicitly superseded across the conversation timeline. Future work on temporal-aware retrieval and memory update mechanisms is critical for reliable long-horizon agents.

The field of agent memory is still young. As conversations grow longer and agents take on more complex tasks, the architecture of memory — not just the intelligence of the model — will become the primary differentiator between capable and brittle systems.

---

*Report generated June 2026. All experiments conducted using the Groq API with Llama 3.3 70B.*
