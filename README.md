# Intelligent Candidate Ranking — Redrob Hackathon

Ranks 100,000 candidate profiles for Redrob's Senior AI/ML Engineer role
(retrieval, ranking, and matching systems) the way a great recruiter would —
by reading career evidence, not counting keywords.

## Reproduce the submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./InnoCoders.csv
```

That's the entire pipeline. **No GPU, no network, no pip installs** — pure
Python standard library (Python ≥ 3.9). Runs in **~31 seconds** on a single
CPU core with < 1 GB of memory, well inside the 5-minute / 16 GB / CPU-only
budget.

Validate the output against the challenge spec:

```bash
python validate_submission.py InnoCoders.csv
```

## How it works

Every candidate gets a score from five weighted components, derived directly
from the JD's stated requirements and stated disqualifiers:

| Component | Weight | What it measures |
|---|---|---|
| Technical depth | 28% | Career-description keyword evidence (45% of this score — the hardest signal to fake), skills list cross-validated against that evidence, platform skill assessments |
| Title / role relevance | 25% | Four-tier title taxonomy (ML/NLP/Search/Reco engineer → data scientist → generic tech → non-technical), current title weighted 65%, career-peak 35% |
| Career quality | 20% | Product-company and tech-industry tenure, penalises consulting-only careers, rewards stability |
| Experience fit | 17% | YOE curve peaking at 6–8 years, recency of ML titles, production-deployment evidence |
| Behavioural availability | 10% | Open-to-work flag, activity recency, recruiter response rate, notice period, GitHub activity |

Location acts as a soft multiplier (×0.70–×1.00, Pune/Noida highest), and
JD-stated disqualifiers stack as hard multiplicative penalties (non-technical
title ×0.25, consulting-only career ×0.50, ghost candidate ×0.60, CV/speech
specialist without NLP crossover ×0.55, research-only career without
production work ×0.50, and more).

### Keyword-stuffer defense

The skills list is self-reported and easy to game, so it is cross-validated
against career descriptions: a candidate claiming advanced FAISS whose entire
career text describes React and QA work gets their skill score cut by 75%.
Keyword matching is word-boundary-safe ("rag" never matches inside
"storage"; "ann" never matches inside "planning").

### Honeypot defense

The dataset plants ~80 "subtly impossible" profiles, forced to relevance
tier 0 in the hidden ground truth. `honeypot_flags()` verifies every profile
against itself:

1. **Date arithmetic** — a role's claimed `duration_months` contradicts its
   own start/end dates (e.g. 166 months claimed, started 33 months ago).
2. **Expert-with-zero-use** — 3+ skills at `expert` proficiency with 0 months
   of use.
3. **Experience span** — stated years-of-experience exceeds the span of the
   entire career history by 2+ years.
4. **Technology anachronism** — skill durations that predate the technology's
   public existence (e.g. 94 months of RAG; RAG has existed since May 2020),
   using month-precision release dates for 15 modern AI technologies.

141 profiles are flagged across the pool; zero appear in the final top 100.

### Reasoning column

Each of the 100 rows carries an individually generated 1–2 sentence
justification built only from facts present in that profile (no
hallucinated skills — verified programmatically), with honest concerns
(long notice, low response rate, off-preference location, out-of-band YOE)
and phrasing that varies by candidate and rank band.

## Files

| File | Purpose |
|---|---|
| `rank.py` | The entire ranking system |
| `InnoCoders.csv` | Ranked top-100 output |
| `validate_submission.py` | Organizer-provided format validator |
| `submission_metadata.yaml` | Submission metadata (mirrors portal entry) |
| `approach_deck.pdf` | Approach explanation deck |
| `requirements.txt` | Empty by design — stdlib only |

## Design constraints honored

- **≤ 5 min, CPU-only, no network**: ~31 s, single core, zero API calls.
- **Deterministic**: same input → byte-identical output, verified by
  running twice and diffing.
- **Tie-break rule**: equal scores ordered by `candidate_id` ascending.
- **Honeypot rate**: 0% in top 100 (disqualification threshold is 10%).
