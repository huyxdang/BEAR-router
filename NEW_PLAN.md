# Project Brief: Compression-Aware LLM Router for The Token Company

## 1. Project Context & Objectives

**Author Context:** I am completing a work trial for **The Token Company**. Their core product is a text compression middleware model called `bear-1.2`, which strips redundant tokens from user prompts to reduce context bloat and lower inference costs before sending the prompt to an LLM.

**My Task:** Design a dynamic LLM router that sits *after* the `bear-1.2` compression step. The router must handle a dynamic pool of models (users can add/remove models on the fly) and enforce user-defined constraints—both hard price caps and explicit model inclusion/exclusion preferences.

**Core Design Hypothesis:** Standard LLM routers assume raw text. I am designing a **Compression-Aware Router** adapted from the *UniRoute* framework. Instead of treating compression and model selection as two sequential decisions, the architecture flattens the decision space: every `[LLM × Compression Tier]` combination is treated as a distinct **Virtual Model** in the routing pool.

**Goal of This Work Trial:** Demonstrate that this architecture is sound and competitive, with a justified model pool, executable experiments, and comparisons against the best single-model baseline (GPT-5.4) and a production routing gateway (OpenRouter).

---

## 2. System Architecture: K-Means Cluster Routing Pipeline

Standard benchmark-based routing fails here because a model's general intelligence does not perfectly correlate with its ability to comprehend heavily token-stripped, compressed text. The architecture uses UniRoute's **K-means** variant — fully unsupervised, no separate training LLMs required.

**Phase 1 (Offline — done once):**

1. **Cluster Training Prompts:** Embed training prompts using a frozen text encoder (OpenAI `text-embedding-3-small`), then run K-means to find $K$ centroids. Training prompts only need embeddings — no model inference, no labels.
2. **Partition the Validation Set:** Assign each prompt in $S_{val}$ to its nearest centroid, forming $K$ representative clusters.
3. **Profile Each Virtual Model:** For every Virtual Model in the pool, run it on the $S_{val}$ prompts (which have ground-truth labels) and compute its average error on each of the $K$ clusters. This produces a $K$-dimensional feature vector ($\Psi$) per Virtual Model — its "compression-tolerance fingerprint."

**Phase 2 (Online — per request):**

1. **Ingestion & Compression:** The raw prompt is processed by `bear-1.2` at the applicable compression tier.
2. **Cluster Assignment:** The compressed prompt is embedded and assigned to the nearest K-means centroid.
3. **Constraint Filtering:** The system filters out any Virtual Model that (a) exceeds the user's hard budget cap, or (b) belongs to a provider the user has explicitly excluded.
4. **Routing via Lookup:** From the surviving candidates, the router selects the Virtual Model with the lowest cost-adjusted average error ($\Psi$) for that cluster.
5. **Execution:** The prompt is sent to the selected Virtual Model.

---

## 3. The "Virtual Model" Candidate Pool

### 3.1 Pool Construction

Each base LLM paired with each compression tier forms a distinct "Virtual Model." With 6 base models and 5 compression tiers (0%, 20%, 40%, 60%, 80%), the pool expands to **30 Virtual Models**:

| Base Model      | 0% | 20% | 40% | 60% | 80% |
| --------------- | -- | --- | --- | --- | --- |
| GPT-5.4         | ✓  | ✓   | ✓   | ✓   | ✓   |
| GPT-4o          | ✓  | ✓   | ✓   | ✓   | ✓   |
| GPT-4o-mini     | ✓  | ✓   | ✓   | ✓   | ✓   |
| GPT-4o-nano     | ✓  | ✓   | ✓   | ✓   | ✓   |
| Claude Sonnet   | ✓  | ✓   | ✓   | ✓   | ✓   |
| Claude Haiku    | ✓  | ✓   | ✓   | ✓   | ✓   |

### 3.2 Model Pool: API-Only, Multi-Tier

All models in the pool are accessed via API. The pool spans three cost tiers across two providers (OpenAI, Anthropic), giving the router a wide cost-capability surface to exploit.

| Tier     | Model            | Provider  | Rationale                                                                |
| -------- | ---------------- | --------- | ------------------------------------------------------------------------ |
| Frontier | GPT-5.4          | OpenAI    | Highest capability ceiling; the "best single model" upper bound          |
| Frontier | Claude Sonnet    | Anthropic | Cross-provider frontier; tests whether compression tolerance varies by provider |
| Mid      | GPT-4o           | OpenAI    | Strong general-purpose model at moderate cost                            |
| Mid      | Claude Haiku     | Anthropic | Fast, cheap Anthropic option; tests small-model compression resilience   |
| Budget   | GPT-4o-mini      | OpenAI    | Low-cost OpenAI option; high volume routing target                       |
| Budget   | GPT-4o-nano      | OpenAI    | Cheapest in pool; tests the floor of acceptable compression-degraded output |

**Why this pool?** It covers a ~100x cost range from nano to frontier, includes two providers to test cross-architecture compression tolerance, and every model is API-accessible — no self-hosting required. The 6-model × 5-tier structure produces 30 Virtual Models, which provides a fine-grained routing surface while remaining executable.

### 3.3 Dynamic Profiling (Calibration)

When a new LLM is added to the platform, it is profiled by running it on $S_{val}$ prompts compressed by `bear-1.2` at each of the 5 tiers (0%, 20%, 40%, 60%, 80%). For each of the $K$ clusters, we compute the model's average error, producing a $K$-dimensional $\Psi$ vector per Virtual Model. Because this only requires inference on $S_{val}$ (no gradient updates, no router retraining), new models can be onboarded dynamically at test time.

---

## 4. User Constraint System

Users have two independent constraint mechanisms, both applied as a **post-training filter** on the candidate pool before the router makes its final selection:

### 4.1 Hard Price Cap

The user sets a maximum effective cost (e.g., $5/1M tokens). The effective cost of a Virtual Model is:

$$c_{\text{eff}}(h) = c_{\text{base}}(h) \times (1 - r)$$

where $r$ is the compression ratio applied by `bear-1.2`. Any Virtual Model where $c_{\text{eff}}(h) > \text{cap}$ is dropped from the candidate pool before routing.

**Example:** A frontier model at $10/1M base cost with 60% compression → $4 effective cost → survives a $5 cap. The same model at 0% compression → $10 → filtered out.

### 4.2 Model Exclusion (Provider Preferences)

Users can explicitly exclude specific models or providers from their candidate pool (e.g., "no OpenAI models"). This is a hard filter applied alongside the price cap. The router only selects from Virtual Models that survive *both* filters.

**Combined Filter Logic:**

$$\mathcal{H}_{\text{active}} = \{ h \in \mathcal{H} \mid c_{\text{eff}}(h) \leq \text{cap} \;\wedge\; \text{provider}(h) \notin \text{exclusion\_list} \}$$

This means the router's effective pool can vary dramatically per-user, which is fine—the router still looks up the $\Psi$ vectors of all surviving candidates and picks the one with the lowest cost-adjusted error for the assigned cluster.

---

## 5. Experimental Design & Validation

### 5.1 Data Pipeline: One Unified Router

We train a single unified router across all three benchmark domains. Following the UniRoute paper's methodology, we pool all prompts from our 3 benchmarks (~6,000 total: ~2,000 from each of FinanceBench, SQuAD 2.0, and CoQA) and randomly split them:

| Split          | % of data | ~Prompts | What it's used for                                        | Needs labels? |
| -------------- | --------- | -------- | --------------------------------------------------------- | ------------- |
| **Training**   | 60%       | ~3,600   | K-means clustering (only needs prompt embeddings)          | No            |
| **Validation** | 10%       | ~600     | Profiling each Virtual Model's per-cluster error ($\Psi$) | Yes           |
| **Test**       | 30%       | ~1,800   | Final evaluation of routing quality                        | Yes           |

**Key details:**

- Training prompts are drawn from all 3 benchmarks (FinanceBench, SQuAD 2.0, CoQA), pooled together. This gives the K-means clusters natural structure across financial reasoning, extractive comprehension, and conversational QA domains.
- Validation prompts ($S_{val}$, ~600) are a separate, non-overlapping split from the same pooled benchmarks. Each Virtual Model is run on these prompts (compressed at the relevant tier) and scored against ground truth to produce its $\Psi$ vector. At ~600 validation prompts and $K_{max}=64$, that's ~9 prompts per cluster minimum — sufficient for stable $\Psi$ vectors.
- Test prompts (~1,800) are held out entirely and used only for final deferral curve evaluation.
- Models are **only ever run on $S_{val}$** during the offline phase. They never touch training prompts.

### 5.2 Baselines

| Baseline                        | What It Proves                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| **GPT-5.4 Only (No Router)**    | The "just use the best model" ceiling. If the router can't beat always-calling GPT-5.4 at lower cost, the architecture has no value. |
| **OpenRouter (Same Pool)**      | Production routing gateway, **restricted to the same 6 models** as our pool. Apples-to-apples comparison — same models, same prompts, but OpenRouter has no awareness of `bear-1.2` compression. |
| **UniRoute (No Compression)**   | Our own architecture, but restricting the pool to 0% compression Virtual Models only (6 VMs instead of 30). This isolates the exact value `bear-1.2` adds on top of the routing. |

> **Why these?** GPT-5.4 Only is the simplest possible strategy — no routing, no compression, just the best model. If we can't match its quality at lower cost, nothing else matters. OpenRouter is a production routing gateway restricted to the same model pool for a fair head-to-head. The No Compression ablation is the critical control that isolates `bear-1.2`'s contribution from the routing logic itself.

### 5.3 The 3 Core Benchmarks

We use the same benchmark families that The Token Company already validates `bear-1.2` against, but scaled up and evaluated across multiple models (where TTC only tested single LLMs). This makes the router directly relevant to their product while testing something they've never done — multi-model routing under compression.

1. **FinanceBench (Financial QA):** ~2,000 questions from real SEC filings. Tests whether models can extract data and reason about dense financial documents under compression. TTC showed compression can *improve* accuracy here — the router should learn to exploit that.
2. **SQuAD 2.0 (Reading Comprehension):** ~2,000 questions including answerable and unanswerable. Tests extractive comprehension — can the model find the right span, or correctly abstain? TTC showed this is more sensitive to heavy compression than financial QA.
3. **CoQA (Conversational QA):** ~2,000 multi-turn questions across multiple domains. Tests context tracking across conversation turns under compression. TTC showed light compression preserves multi-turn accuracy perfectly.

**Total: ~6,000 prompts pooled across all 3 benchmarks.**

### 5.4 Tuning Protocol: The Two Axes

The router's performance is governed by two independent tuning axes. We sweep each to find the optimal configuration.

**Tuning Objective:** Following the UniRoute paper, we select hyperparameters by maximizing the **area under the deferral curve (AUC)** on the validation set. The deferral curve plots accuracy vs. cost as we sweep the cost-quality tradeoff parameter $\lambda$ from 0 to $\infty$. Higher AUC = better accuracy per dollar spent. This is our single optimization target for tuning.

#### Axis 1: Number of Clusters ($K$)

K-means partitions the prompt embedding space into $K$ clusters. Too few → the router under-segments and fails to specialize. Too many → sparse clusters with unreliable performance estimates.

| Parameter | Sweep Range        | Granularity         |
| --------- | ------------------ | ------------------- |
| $K$       | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 | 10 settings          |

**Metric:** AUC of the deferral curve on the validation set. Select $K$ that maximizes AUC.

#### Axis 2: Validation Set Size ($|S_{val}|$)

The validation set is what profiles each Virtual Model's compression tolerance. Too small → noisy $\Psi$ vectors, unreliable routing. Too large → higher profiling cost when onboarding new models.

| Parameter    | Sweep Range                  | Granularity          |
| ------------ | ---------------------------- | -------------------- |
| $|S_{val}|$  | 25, 50, 100, 200, 500       | 5 settings           |

**Metric:** AUC of the deferral curve on the validation set, plus stability of $\Psi$ vectors across random seeds (variance of per-cluster accuracy).

#### Sweep Execution Plan

Full grid search: $5 \times 5 = 25$ configurations, evaluated on the unified test set. **Total: 25 evaluation runs.** Directly executable.

---

## 6. Deliverables Summary

| Deliverable                     | What It Contains                                                       |
| ------------------------------- | ---------------------------------------------------------------------- |
| **Deferral Curve**              | Cost vs. Accuracy Pareto frontier: our system vs. GPT-5.4 Only, OpenRouter (same pool), and our own no-compression ablation |
| **Tuning Sensitivity Report**   | Per-axis sweep results showing how $K$ and $|S_{val}|$ affect AUC of the deferral curve |
| **Compression Value Isolation** | Direct comparison of UniRoute (with compression) vs. UniRoute (without) to quantify `bear-1.2`'s contribution |

---

## Open Questions

1. **OpenRouter API Access:** Do we have confirmed access to OpenRouter's API? Need to verify that we can restrict it to our exact 6-model pool for a fair comparison.

-- OpenRouter API Key is called OPENROUTER_API_KEY 
