# AI Workflows and Publishing in Finance
## HKUST PhD Short Course — 3 Hours
**Instructor:** J. Anthony Cookson  
**Date:** April 23, 2026  
**Format:** Structured discussion with live demos  
**Wiki reference:** https://velikov-mihail.github.io/ai-econ-wiki/ (Mihail Velikov, Penn State Smeal)

---

## Course Goals

By the end of this session, students should be able to:
1. Set up a functional AI workflow for empirical finance research — not just as a chatbot, but as a persistent assistant
2. Use AI to compress the gap between a dataset and a research result
3. Understand LLMs as *measurement tools*, not just productivity aids — with concrete application to asset pricing and investor behavior
4. Reason clearly about how AI is changing academic publishing and what that means for careers starting now

---

## Structure (3 hours)

| Block | Topic | Time |
|---|---|---|
| 1 | From Skeptic to Power User: AI as a Research Assistant | 50 min |
| 2 | From Empty Folder to Research Result: AI in Empirical Workflows | 70 min |
| 3 | LLMs as Measurement Tools + The Future of Finance Research | 60 min |

---

## Block 1 — From Skeptic to Power User: AI as a Research Assistant (50 min)

### 1.1 Motivation: What Changed, and Why Now (10 min)

The shift is not that AI got smarter at writing. It's that AI can now *act* — read files, run code, manage memory, call external services — inside a project folder that persists across sessions. That's what makes it a research assistant rather than a chatbot.

Two vivid illustrations:
- A finance paper written in four days (vibe research) — not endorsing this, but it shows the frontier
- A political economist (Chris Blattman, UChicago) builds a full executive assistant with no prior coding experience: email triage, project dashboards, calendar, expenses

Key framing: **"Become a complement to these tools, not a substitute."** The goal is not to automate your research — it's to eliminate the friction between your judgment and a result.

### 1.2 The Blattman Framework (15 min)

**Site: https://claudeblattman.com** — Blattman documented his entire workflow publicly as it evolved.

Core methodology: **Prompt → Plan → Review → Revise**
- Start with a task description
- Let the AI propose a plan before acting
- Review the plan — this is where your judgment matters
- Revise and iterate

Key insight: the value is not in any single output but in the **persistent context** — a CLAUDE.md file that tells the assistant who you are, what your projects are, how you want to work. Like onboarding a very fast RA who never forgets what you've told them.

What Blattman built (and what I built following a similar path):
- `/checkin` — daily inbox triage, calendar, task list in one pass
- `/expenses` — reimbursement tracking from forwarded receipts
- `/editor` — editorial workload dashboard (manuscripts, deadlines, overdue items)

**Live demo: show `/checkin` or `/editor` in action** — brief, but concrete. The goal is to make the setup feel achievable, not magical.

*References:* claudeblattman.com; Velikov wiki: "Building an AI Executive Assistant"; "Chris Blattman Thread: From Claude Code Skeptic to Power User"

**Discussion prompt:** *What's the most time-consuming non-research task in your week? Write it down — we'll come back to it.*

### 1.3 The Jagged Frontier: Where AI Helps and Where It Misleads (10 min)

AI is not uniformly better or worse than a human — it's *jagged*. Strong at code generation, reformatting, first drafts, summarization. Unreliable at fact retrieval, domain-specific causal reasoning, citation accuracy.

For finance PhD students specifically:
- **Trust:** boilerplate code, data cleaning, literature summaries from documents you provide
- **Verify:** any specific number, citation, or empirical claim AI generates
- **Don't delegate:** identification strategy, economic interpretation, judgment about what's interesting

The bottleneck is always human verification — AI speeds up drafts but creates new review work.

*References:* Velikov wiki: "The Shape of AI: Jaggedness, Bottlenecks and Salients"; "Sycophancy and Bias in AI"

### 1.4 Setup in Practice (15 min)

Three things that make AI useful vs. a toy:
1. **CLAUDE.md** — persistent context file: who you are, your projects, your preferences, your data sources
2. **Project folders** — AI works best inside a structured folder with data, code, and notes co-located
3. **Skills/commands** — reusable workflows (like `/checkin`) that encode your recurring tasks

*References:* Velikov wiki: "AI Project Folders"; "A Real CLAUDE.md"; "Getting Started with Claude Code: A Researcher's Setup Guide"; PGP Episode 1 notes (paulgp.substack.com)

**Discussion prompt:** *If you were writing a CLAUDE.md for yourself right now — a one-page brief for an AI assistant — what would be in it?*

---

## Block 2 — From Empty Folder to Research Result: AI in Empirical Workflows (70 min)

*This block is anchored in Paul Goldsmith-Pinkham's mini-series (Markus' Academy / BCF Princeton, March 2026): https://bcf.princeton.edu/events/paul-goldsmith-pinkham-mini-series-on-claude-code-for-applied-economists/*

### 2.1 The Core Demonstration: Empty Folder → Figure (20 min)

PGP's key insight: AI "dramatically shrinks the gap between a vague research idea and initial results." The workflow:
1. Start with a question and an empty folder
2. Describe the data you need and where to get it
3. Let AI plan the pipeline, then iterate
4. Review outputs at each step — your judgment is the quality control

**Demo — Social Signal data, live in class:**

Dataset: publicly available firm-day social signal index from "The Social Signal"
(Cookson, Lu, Mullins & Niessner, JFE 2024) — https://data.mendeley.com/datasets/xffyybvw4j/1.
821,534 firm-day observations, 1,500 firms, 2012–2021. Variables: cross-platform
sentiment PC1 and attention PC1, both z-scored.

The demo runs in three layers of increasing complexity — each driven by a
single plain-English prompt:

**Layer 1 — Firm-level basics** (`demo_script.py`)

Prompt: *"I have a firm-day social signal dataset. Help me understand what's
in it and produce a time-series figure of average sentiment and attention.
Then merge with Yahoo Finance returns and test whether sentiment predicts
next-day abnormal returns."*

Produces: time-series plot with GME annotation, distribution figures,
return predictability by decile, OLS with correct signs (sentiment +, attention −).

**Layer 2 — Market-level analysis** (`market_analysis.py`)

Prompt: *"Can you do a demo of how you would produce an aggregate index and
explore its market-level predictive properties for trading, volatility, and
returns? Also consider the reverse relation — how do recent trading, vol and
returns predict this aggregate index? Does this have anything to do with
market news events like FOMC dates?"*

Produces: equal-weighted daily market index, forward and reverse predictive
regressions with Newey-West SEs, FOMC event study (±15-day window, cumulative
return), leads-and-lags cumulative return plot (high vs. low signal days,
t−6 baseline → t+15).

Key findings: high aggregate sentiment predicts higher next-day VIX (β = +0.46,
t = 2.73) and volume; rising VIX dampens sentiment within one day; signal
shifts around FOMC announcements.

**Layer 3 — Cap-weighted robustness** (`market_analysis_capweighted.py`)

Prompt: *"Can you aggregate in a market value way and redo the analysis?"*

Produces: comparison of EW-1500 vs. CW-8 (large-cap-tilted) signal. Sentiment
correlation r = 0.39 — large caps diverge meaningfully from equal-weighted
breadth. Cap-weighting sharpens return predictability (CW β = −0.0007, t = −2.13*).

**Iteration on display** — prompt used to refine the event-study visualization:

> *"Can you show the return plots as cumulative plots from a benchmark return
> date of t−6 through t+15?"*
> *(Correction)* *"I meant to do this in the distributed lag plot. Could be a
> leads and lags plot."*

Point: the researcher supplies the economic intuition about the right
visualization; AI adapts immediately. The "correction" is part of the workflow,
not a failure.

**Narration points:**
- Three layers, four prompts, ~15 minutes of real session time
- Each prompt added economic content that required a researcher judgment
- The code was never the bottleneck — the questions were
- The full demo package (scripts + figures + this prompt log) lives in
  `demos/Market-Level Analysis/demo_guide.md`

*References:* PGP Episode 2 ("From an Empty Folder to a Figure"); Velikov wiki: "From an Empty Folder to a Figure using Claude Code"

### 2.2 Structured Databases, Larger Pipelines, and AI-Augmented Replication (15 min)

When data gets large or messy, structure matters more:
- Converting flat files to Parquet/DuckDB for fast querying (PGP Episode 4)
- Extracting structured data from unstructured sources — e.g., EDGAR 10-K filings (PGP Episode 3)
- Planning mode: ask AI to plan the full pipeline before writing any code
- Sub-agents: breaking a complex task into parallel components

Example from PGP Episode 4: mortgage market panel from HMDA data — county lender concentration, fintech/non-bank growth. This is the kind of data assembly task that used to take a RA a month.

**New angle: AI-augmented replication packages**

The same capability that lets you *build* a new pipeline also lets you
*navigate* an existing one — with no prior knowledge of the codebase.

Concrete example: Dickerson, Julliard & Mueller, "Co-Pricing in the Factor Zoo"
(*Journal of Financial Economics*) — https://github.com/Alexander-M-Dickerson/co-pricing-factor-zoo

This replication package is explicitly designed for AI-assisted replication.
It ships with a `.claude/` directory containing `paper-context.md` and custom skills:

- `/onboard` — validates R installation, packages, and data; auto-fixes gaps
- `/replicate-paper` — runs the full pipeline with automatic error recovery
- `/explain-paper` — explains any table, figure, or method on demand

The user prompt: *"Replicate the main text. If packages or data are missing,
bootstrap them automatically first."* No prior knowledge of the codebase required.

**Why this matters for the course:**
- Replication is currently a bottleneck in empirical finance — packages are
  complex and under-documented
- AI lowers the cost of replication to near zero *if* the package is designed for it
- This creates an emerging norm: replication packages should include AI context
  files alongside the code. Authors who do this make their work easier to
  build on; editors may start requiring it
- For students: an AI-ready replication package is a new form of research output
  that signals methodological transparency

*References:* PGP Episodes 3 and 4; Velikov wiki: "From EDGAR Filings to a Structured Database"; "Large Datasets and Structured Databases"; "Claude Code WRDS Toolkit"; Dickerson et al. repo (above)

**Discussion prompt:** *Describe a data assembly task in your current project that you've been putting off because it's tedious. How would you describe it to an AI in plain English?*

### 2.3 Stress-Testing Research Designs (15 min)

AI as a pre-mortem tool — identify the fatal flaw before referees do:
- Describe your identification strategy in a paragraph; ask AI to find the holes
- Ask: "What would a skeptical referee say about this?" 
- Ask: "What's the most obvious alternative explanation for this result?"
- Caveat: AI will also find problems that aren't real — calibration matters, and you need domain knowledge to filter

Related: multiple AI agents auditing your DiD code for specification errors before you submit

*References:* Velikov wiki: "Stress-Test Any Plan"; "AI-Powered Pipeline to Stress-Test Research Ideas"; "Claude Code 24: Multiple Agents Auditing Your DiD Code"

**Discussion prompt:** *Describe your current identification strategy in two sentences. What's the hardest question you'd get at a seminar?*

### 2.4 Literature and Feedback (20 min)

- **NotebookLM**: document-grounded AI — you feed it papers, it reasons over them. Much safer than asking AI to recall literature from memory (no hallucination risk on the documents you provide)
- **AI-generated referee reports**: useful as a first-pass signal, not as a substitute for judgment. Good at flagging missing robustness checks, unclear writing, obvious alternative hypotheses
- **Feedback machines**: the iterative loop — draft, AI critique, revise, repeat

The key principle: use AI to find weaknesses, not to write your paper.

*References:* Velikov wiki: "NotebookLM: Document-Grounded AI"; "Feedback Machines"; "AI Research Feedback Skills"; Refine.ink

---

## Block 3 — LLMs as Measurement Tools + The Future of Finance Research (60 min)

### 3.1 Beyond the Assistant: LLMs as Research Instruments (25 min)

**The Market's Mirror** (Cookson, Bhagwat, Dim, Niessner) — a case study in using LLMs not as writing aids but as *measurement tools*.

The question: How do demographics drive investor disagreement? And does demographic disagreement predict trading?

The approach:
- Imbue Llama 3.1 8B with 216 investor personas (72 demographic combinations × 3 political orientations, weighted by FINRA survey data)
- Ask all 216 personas for buy/hold/sell sentiment on 5.5 million S&P500 news headlines (2010–2025)
- Compute weighted SD of sentiment = LLM disagreement measure

Why this matters methodologically:
- A *survey that would not be possible with humans* — no person would fill in 1.188 billion items
- Allows revision and experimentation: when you update the paper, you re-query the same way
- Fixed training window (Dec 2023) → post-training validation (2024–2025)

Key findings:
- Income and politics drive disagreement; social/soft news generates more disagreement than hard financial news
- LLM disagreement validates against human survey data (Iliewa et al. 2025, Toubia et al. 2025 digital twins)
- Disagreement predicts abnormal trading volume — especially retail trading

**The broader point:** LLMs are not just productivity tools. They are instruments for measuring constructs — beliefs, sentiment, preferences, reasoning — that we couldn't measure at scale before. This opens an entire new research agenda for finance.

*Reference:* Guest lecture slides; Velikov wiki: "Generative AI for Economic Research: Use Cases and Implications" (Korinek)

**Discussion prompt:** *What latent construct in your research area would you want to measure at scale if you could run 1 billion surveys? What would the right instrument look like?*

### 3.2 How Publishing Is Changing (20 min)

This is the part that matters most for a PhD student's career trajectory.

- **"Research and publishing are now two different things"** — the pipeline is decoupling. AI lowers the cost of producing a plausible-looking paper; it does not lower the cost of producing a *true finding*.
- AI one-shot papers, vibe research, automated policy evaluation (Project APE): these exist and will get better
- The zero profit condition: if anyone can produce a passable paper cheaply, what's the marginal value of one more?

**What this means for PhD students:**
- The premium will shift toward: deep domain knowledge, genuine identification creativity, research questions that require real-world access (proprietary data, field experiments, regulatory relationships)
- Measurement novelty — like the LLM persona approach — is one source of durable edge
- Replication and robustness become *easier* to demand; editors will raise the bar

**Replication packages as a new publication norm:**

Tie back to the Dickerson et al. example from Block 2. If AI can replicate a
paper in one command, editors will start requiring AI-ready replication packages
— not just code dumps but packages with context files, skills, and
self-documenting pipelines. This is already happening at the frontier.

For a PhD student, this cuts two ways:
- *More accountability:* your identification strategy and code are legible to
  anyone with Claude Code — reviewers, competitors, journal editors
- *More leverage:* you can build on others' work faster; a well-documented
  replication package is itself a contribution that gets cited and used

The zero-profit condition applies to cheap papers, not to well-documented,
AI-navigable research packages that make others' work easier to build on.

The philosophical question: *"Hadn't the satisfaction always been in the discovering, not the discoveries?"* The research process may change faster than our reasons for doing research.

*References:* Velikov wiki: "Claude Code 27: Research and Publishing Are Now Two Different Things"; "The Zero Profit Condition Is Coming"; "AI One-Shot Papers"; "Vibe Research"; "When Will the Research Paper Disappear in Economics?"; Dickerson et al. repo (Block 2.2)

**Discussion prompt:** *What are you building that AI can't replicate? And what part of your current workflow are you doing manually that AI could handle tomorrow?*

### 3.3 Closing: Open Questions (15 min)

No settled answers — genuine uncertainty about:
- How fast autonomous research agents arrive and how good they get
- Whether LLM measurement tools will be accepted as identification or dismissed as fancy text mining
- What happens to the economics PhD as a credential if AI compresses the time-to-paper
- Where the satisfaction in research comes from once discovery is cheap

*References:* Velikov wiki: "The Train Has Left the Station: Agentic AI and the Future of Social Science"; "Something Big Is Happening"; "The Bitter Lesson"

**Closing prompt:** *What's one thing you'll do differently in your research workflow this week? And what's the question about AI and research that you most want answered?*

---

## Demo Script (Block 2.1) — Social Signal Data

**Setup:** Download dataset from https://data.mendeley.com/datasets/xffyybvw4j/1  
**Tool:** Claude Code in a fresh project folder  
**Sequence:**
1. Drop the data files into the folder
2. Prompt: "I have StockTwits sentiment data and return data merged at the firm-day level. Help me understand what's in this dataset and produce a summary figure showing average sentiment over time."
3. Walk through: AI reads the data dictionary, proposes a plan, writes the merge/summarize code, debugs, produces figure
4. Extend: "Now regress abnormal returns on sentiment from the prior day, controlling for firm and time fixed effects"
5. Commentary: point out where the researcher's judgment is critical (variable definitions, sample restrictions, what controls matter)

**Key pedagogical points to narrate:**
- AI is fast on the mechanical parts; you supply the economic question
- The iteration loop (AI proposes → you review → AI revises) is the workflow
- You could do this with your own data from day one of a PhD program

---

## Readings / Prep (optional)

**Conceptual:**
- "The Shape of AI: Jaggedness, Bottlenecks and Salients" (Velikov wiki)
- "Research and Publishing Are Now Two Different Things" (Velikov wiki)
- "The Bitter Lesson" — Rich Sutton

**Practical:**
- PGP Episode 1 notes: https://paulgp.substack.com/p/getting-started-with-claude-code
- PGP Episode 2 notes: https://paulgp.substack.com/p/from-an-empty-folder-to-a-figure
- claudeblattman.com — browse the workflows

**For the ambitious:**
- Korinek, "Generative AI for Economic Research: Use Cases and Implications for Economists"
- "Feedback Machines: Writing and Editing Research Papers with Generative AI"

---

## To Build Out

- [x] Prepare Social Signal demo — tested end-to-end; see `demos/Social Signal Demo/` and `demos/Market-Level Analysis/`
- [x] Market-level analysis demo — three layers (EW, cap-weighted, leads-and-lags); prompt log in `demo_guide.md`
- [ ] Decide whether students install Claude Code before class (recommended: yes, with setup guide sent in advance)
- [ ] Pull 2–3 specific slides from guest_lecture_cookson.pptx as anchors for Block 3.1 (Market's Mirror)
- [ ] Consider: ask students to bring a 1-paragraph research design to stress-test live in Block 2.3
- [ ] Build slide deck once outline is settled (Block 3 naturally follows guest lecture structure)
- [ ] Note: second lecture (April 27) covers AI in finance research more deeply — this course is the workflow foundation
