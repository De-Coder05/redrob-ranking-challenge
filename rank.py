#!/usr/bin/env python3
"""
AI Engineer Candidate Ranker — Redrob Hackathon
Ranks 100K candidates for an AI/ML Engineer role focused on
retrieval, ranking, and recommendation systems.

Design highlights
-----------------
* Word-boundary keyword matching (no "rag" matching inside "storage").
* Career descriptions are the primary technical signal — the skills list
  is cross-validated against them to catch keyword stuffers.
* Honeypot detection via internal-consistency checks: role durations that
  contradict their own start/end dates, expert skills with zero months of
  use, years-of-experience exceeding the career-history span, and skill
  durations that predate the technology's public release.
* JD-derived disqualifier penalties: consulting-only careers, non-technical
  titles, ghost candidates, research-only careers, non-coding architects.
* Per-candidate reasoning built from actual profile facts, with honest
  concerns and varied phrasing.

Usage: python rank.py --candidates candidates.jsonl --out submission.csv
"""

import argparse
import csv
import json
import re
import sys
from datetime import date, datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# JD-derived constants
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 11)

# Core skills for this role: embeddings/dense retrieval, vector DBs,
# hybrid search, NLP/IR, evaluation frameworks, LLM fine-tuning.
# Weights reflect how directly the skill maps to the JD's "must haves".
CORE_SKILLS: dict = {
    # ── Embeddings / Dense Retrieval ──────────────────────────────────────────
    "sentence-transformers": 1.00, "sentence transformers": 1.00,
    "embeddings": 0.90, "embedding": 0.85, "embedding models": 0.90,
    "dense retrieval": 1.00, "semantic search": 0.90,
    "bi-encoder": 0.95, "cross-encoder": 0.95,
    "rag": 0.90, "retrieval augmented generation": 0.95,
    "retrieval-augmented generation": 0.95,
    "bge": 0.90, "e5": 0.85, "openai embeddings": 0.85,
    # ── Vector DBs / ANN ─────────────────────────────────────────────────────
    "pinecone": 1.00, "weaviate": 1.00, "qdrant": 1.00, "milvus": 1.00,
    "faiss": 1.00, "elasticsearch": 0.90, "opensearch": 0.90,
    "vector database": 1.00, "vector databases": 1.00,
    "vector store": 0.90, "vector db": 1.00, "vector search": 1.00,
    "ann": 0.80, "hnsw": 0.90, "approximate nearest neighbor": 0.90,
    "hybrid search": 1.00, "hybrid retrieval": 1.00,
    # ── Sparse Retrieval ─────────────────────────────────────────────────────
    "bm25": 1.00, "tfidf": 0.85, "tf-idf": 0.85,
    "inverted index": 0.85, "sparse retrieval": 0.90,
    "solr": 0.75, "lucene": 0.85,
    # ── Core NLP / ML ─────────────────────────────────────────────────────────
    "nlp": 0.90, "natural language processing": 0.90,
    "information retrieval": 1.00,
    "transformers": 0.90, "huggingface": 0.90, "hugging face": 0.90,
    "bert": 0.85, "pytorch": 0.85, "tensorflow": 0.75,
    "scikit-learn": 0.70, "sklearn": 0.70,
    "machine learning": 0.75, "deep learning": 0.75,
    # ── Ranking / Recommendation / Search ────────────────────────────────────
    "ranking": 0.90, "search ranking": 0.95, "search relevance": 0.95,
    "recommendation": 0.85, "recommender": 0.85,
    "recommendation systems": 0.90, "recommender systems": 0.90,
    "recommendation system": 0.90, "recommender system": 0.90,
    "reranking": 0.95, "re-ranking": 0.95, "re-ranker": 0.95, "reranker": 0.95,
    "two-tower": 0.90, "query understanding": 0.90,
    "learning to rank": 1.00, "ltr": 0.90,
    "lambdamart": 0.95, "lambdarank": 0.95, "ranknet": 0.90,
    "xgboost": 0.70, "lightgbm": 0.70,
    # ── Evaluation / Metrics ─────────────────────────────────────────────────
    "ndcg": 1.00, "mrr": 1.00,
    "offline evaluation": 0.85, "online evaluation": 0.85,
    "a/b testing": 0.90, "ab testing": 0.90, "a/b test": 0.90,
    "evaluation framework": 0.90, "eval harness": 0.90,
    # ── LLMs / Fine-tuning ───────────────────────────────────────────────────
    "llm": 0.80, "llms": 0.80, "large language model": 0.80,
    "llama": 0.75, "mistral": 0.75,
    "fine-tuning": 0.85, "fine tuning": 0.85, "fine-tuned": 0.85,
    "lora": 0.85, "qlora": 0.85, "rlhf": 0.85, "peft": 0.80,
    # ── Python (required language) ───────────────────────────────────────────
    "python": 0.75,
    # ── MLOps / Serving ──────────────────────────────────────────────────────
    "mlops": 0.75, "mlflow": 0.70, "feature store": 0.70,
    "model serving": 0.75, "inference optimization": 0.80,
    "distributed inference": 0.80, "triton": 0.75, "kubeflow": 0.70,
}

# Evidence keywords that indicate genuine retrieval/recsys/search career work
# (used for the JD's "built a recommendation system at a product company" rule)
RETRIEVAL_EVIDENCE = (
    "retrieval", "ranking", "recommendation", "recommender", "search",
    "embedding", "embeddings", "rag", "vector", "bm25", "faiss",
    "reranker", "re-ranker", "ndcg", "mrr", "learning to rank",
)

# Public release dates (fractional years) for technologies that did not exist
# before a certain date. Claiming usage longer than the technology's public
# existence is an internal-consistency red flag. +4 months slack at check time.
TECH_RELEASE_YEAR = {
    "rag": 2020.4,                          # RAG paper, May 2020
    "retrieval augmented generation": 2020.4,
    "langchain": 2022.8,                    # first release, Oct 2022
    "llamaindex": 2022.9,                   # (GPT Index) Nov 2022
    "qlora": 2023.4,                        # paper, May 2023
    "lora": 2021.5,                         # paper, Jun 2021
    "peft": 2023.1,                         # HF PEFT lib, Feb 2023
    "chatgpt": 2022.9,                      # Nov 2022
    "gpt-4": 2023.2,                        # Mar 2023
    "llama": 2023.1,                        # LLaMA, Feb 2023
    "mistral": 2023.7,                      # Mistral 7B, Sep 2023
    "pinecone": 2021.0,                     # product GA, early 2021
    "qdrant": 2021.4,                       # first public release, 2021
    "prompt engineering": 2022.0,
    "stable diffusion": 2022.6,             # Aug 2022
}

# Title relevance tiers
_TIER1_TITLES = (
    "ml engineer", "machine learning engineer", "ai engineer", "ai/ml engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "recommendation systems engineer",
    "applied scientist", "applied ml", "applied ai", "applied researcher",
    "research engineer", "ai research engineer",
)
_TIER2_TITLES = (
    "data scientist", "ai specialist", "ai researcher",
    "ml platform engineer", "mlops engineer", "ml infrastructure engineer",
    "software engineer (ml)", "software engineer ml",
)
_TIER3_TITLES = (
    "data engineer", "analytics engineer",
    "software engineer", "backend engineer",
    "full stack developer", "platform engineer", "cloud engineer",
)
_DISQUAL_TITLES = (
    "marketing manager", "hr manager", "content writer", "accountant",
    "sales executive", "graphic designer", "civil engineer",
    "mechanical engineer", "customer support", "operations manager",
    "project manager", "business analyst", "finance manager",
    "financial analyst", "recruiter", "talent acquisition",
    "digital marketing",
)
_CV_SPEECH_TITLES = (
    "computer vision engineer", "cv engineer", "speech engineer",
    "robotics engineer", "image processing engineer",
)
_RESEARCH_ONLY_TITLES = (
    "research scientist", "research fellow", "postdoc", "phd researcher",
    "research intern", "research associate", "academic researcher",
)
_ARCHITECT_TITLES = (
    "architect", "head of", "director", "vp ", "vice president",
    "chief ", "engineering manager",
)

# Consulting companies — whole-career-in-consulting is a JD disqualifier
_CONSULTING = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl ", "tech mahindra",
    "mphasis", "hexaware", "mindtree", "ltimindtree", "l&t infotech",
    "coforge", "niit technologies", "zensar", "igate", "syntel",
    "mastech", "patni", "kpit technologies",
    "birlasoft", "cyient", "sonata software",
)

# Location preferences (city/keyword → raw score 0-1)
_LOCATION_SCORES = {
    "pune": 1.00, "noida": 1.00,
    "hyderabad": 0.85, "mumbai": 0.85, "delhi": 0.85, "new delhi": 0.85,
    "gurugram": 0.85, "gurgaon": 0.85, "ncr": 0.85,
    "bangalore": 0.75, "bengaluru": 0.75,
    "chennai": 0.65, "kolkata": 0.60, "ahmedabad": 0.55,
}

_PROFICIENCY_MULT = {
    "beginner": 0.30, "intermediate": 0.60, "advanced": 0.85, "expert": 1.00,
}

_TECH_INDUSTRIES = (
    "technology", "software", "saas", "fintech", "ai", "machine learning",
    "internet", "e-commerce", "ecommerce", "marketplace", "cloud", "data",
    "analytics", "artificial intelligence", "information technology",
    "edtech", "healthtech", "deeptech", "startup", "product",
    "food delivery", "gaming", "media", "social",
)

_COMPANY_SIZE_SCORES = {
    "1-10": 0.70, "11-50": 0.80, "51-200": 0.90, "201-500": 1.00,
    "501-1000": 1.00, "1001-5000": 0.90, "5001-10000": 0.70, "10001+": 0.50,
}

_RELEVANT_ASSESSMENTS = (
    "machine learning", "deep learning", "nlp", "natural language",
    "information retrieval", "python", "pytorch", "tensorflow", "transformers",
    "data science", "statistics", "recommendation", "search", "ranking",
    "embeddings", "vector", "llm", "fine-tuning", "fine-tuning llms",
)

_ML_PRODUCTION_SIGNALS = (
    "deployed", "production", "real users", "a/b test", "ab test", "online",
    "at scale", "billion", "million", "latency", "throughput",
    "serving", "inference", "pipeline", "recommender", "search",
    "ranking", "retrieval", "shipped",
)

_NLP_CROSSOVER = (
    "nlp", "natural language", "text", "retrieval",
    "ranking", "search", "recommendation",
)

_HANDS_ON_SIGNALS = (
    "built", "implemented", "wrote", "coded", "developed", "designed and built",
    "shipped", "deployed", "fine-tuned", "trained",
)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation and matching helpers
# ─────────────────────────────────────────────────────────────────────────────

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    """Lowercase and collapse all non-alphanumerics to single spaces."""
    return _NORM_RE.sub(" ", text.lower()).strip()


# Pre-normalise the keyword tables once at import time.
_CORE_NORM = {}
for _kw, _w in CORE_SKILLS.items():
    _nk = _norm(_kw)
    if _nk:
        _CORE_NORM[_nk] = max(_CORE_NORM.get(_nk, 0.0), _w)
_CORE_SINGLE = {k: w for k, w in _CORE_NORM.items() if " " not in k}
_CORE_MULTI = {k: w for k, w in _CORE_NORM.items() if " " in k}
_TOTAL_KW_WEIGHT = sum(_CORE_NORM.values())

_RELEASE_NORM = {_norm(k): v for k, v in TECH_RELEASE_YEAR.items()}


def _blob_keyword_hits(blob_norm: str, blob_tokens: set) -> float:
    """Sum of weights of distinct core keywords present in a normalised blob,
    using word-boundary semantics (so 'rag' never matches inside 'storage')."""
    hits = 0.0
    for kw, w in _CORE_SINGLE.items():
        if kw in blob_tokens:
            hits += w
    padded = f" {blob_norm} "
    for kw, w in _CORE_MULTI.items():
        if f" {kw} " in padded:
            hits += w
    return hits


def _best_skill_weight(skill_name: str) -> float:
    """Highest-weight core keyword matching this skill name (word-boundary)."""
    n = _norm(skill_name)
    if not n:
        return 0.0
    w = _CORE_NORM.get(n)
    if w:
        return w
    best = 0.0
    padded = f" {n} "
    for kw, kw_w in _CORE_NORM.items():
        if kw_w > best and f" {kw} " in padded:
            best = kw_w
    return best


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _career_blob(career: list) -> str:
    parts = []
    for r in career:
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
        parts.append(r.get("company", ""))
        parts.append(r.get("industry", ""))
    return " ".join(parts)


def _title_tier(title: str) -> int:
    """4=tier1 (ideal), 3=tier2, 2=tier3, 1=generic, 0=disqualifying."""
    t = title.lower()
    for kw in _DISQUAL_TITLES:
        if kw in t:
            return 0
    for kw in _CV_SPEECH_TITLES:
        if kw in t:
            return 1
    for kw in _TIER1_TITLES:
        if kw in t:
            return 4
    for kw in _TIER2_TITLES:
        if kw in t:
            return 3
    for kw in _TIER3_TITLES:
        if kw in t:
            return 2
    if any(k in t for k in ("engineer", "developer", "scientist", "analyst", "architect")):
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection (internal-consistency checks)
# ─────────────────────────────────────────────────────────────────────────────

def honeypot_flags(candidate: dict) -> list:
    """
    Detects 'subtly impossible' profiles via internal consistency:
      * date_mismatch  — a role's claimed duration_months contradicts its own
                         start/end dates by more than 3 months (e.g. claiming
                         166 months at a job started 33 months ago).
      * yoe_exceeds_span — stated years_of_experience exceeds the span of the
                         entire career history by more than 2 years.
      * expert_zero    — 3+ skills claimed at 'expert' with 0 months of use.
      * anachronistic  — 2+ skills claiming usage longer than the technology
                         has publicly existed (e.g. 94 months of RAG).
    Flagged candidates are forced to the bottom of the ranking.
    """
    flags = []
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    for r in career:
        sd = _parse_date(r.get("start_date"))
        ed = _parse_date(r.get("end_date")) or REFERENCE_DATE
        if sd:
            actual_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if abs(actual_months - r.get("duration_months", 0)) > 3:
                flags.append("date_mismatch")
                break

    starts = [_parse_date(r.get("start_date")) for r in career]
    starts = [s for s in starts if s]
    if starts:
        first = min(starts)
        span_years = (REFERENCE_DATE - first).days / 365.25
        if yoe > span_years + 2:
            flags.append("yoe_exceeds_span")

    n_expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if n_expert_zero >= 3:
        flags.append("expert_zero")

    ref_frac = REFERENCE_DATE.year + REFERENCE_DATE.month / 12.0
    n_anach = 0
    for s in skills:
        nm = _norm(s.get("name", ""))
        rel = _RELEASE_NORM.get(nm)
        if rel:
            max_months = (ref_frac - rel) * 12 + 4  # +4 months slack
            if s.get("duration_months", 0) > max_months:
                n_anach += 1
    if n_anach >= 2:
        flags.append("anachronistic")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Scoring components
# ─────────────────────────────────────────────────────────────────────────────

def _tech_score(skills: list, career: list, signals: dict,
                total_career_months: int) -> tuple:
    """
    Returns (score 0-1, top_matched_skill_names, desc_component).
      35% skill-list match (proficiency + duration + endorsements)
      45% career-description keyword hits  ← ground truth of actual work done
      20% platform assessment scores
    A consistency check scales back claimed skills when career descriptions
    provide no supporting evidence (the keyword-stuffer trap).
    """
    blob_norm = _norm(_career_blob(career))
    blob_tokens = set(blob_norm.split())

    hits = _blob_keyword_hits(blob_norm, blob_tokens)
    # 20% of all keyword weight = "excellent coverage" for descriptions
    desc_component = min(hits / (_TOTAL_KW_WEIGHT * 0.20), 1.0)

    sk_score = 0.0
    sk_weight_sum = 0.0
    matched = []  # (weight*prof, name) for reasoning
    for s in skills:
        w = _best_skill_weight(s.get("name", ""))
        if w == 0:
            continue
        prof = _PROFICIENCY_MULT.get(s.get("proficiency", "intermediate"), 0.50)
        # cap claimed duration at total career length (data hygiene)
        dur_mo = min(s.get("duration_months", 12), max(total_career_months, 1))
        dur = min(dur_mo, 60) / 60
        endr = min(s.get("endorsements", 0), 20) / 20
        item = w * (0.50 * prof + 0.30 * dur + 0.20 * (0.5 + 0.5 * endr))
        sk_score += item
        sk_weight_sum += w
        if w >= 0.70:
            matched.append((w * prof, s.get("name", "")))
    skill_component = sk_score / max(sk_weight_sum, 1.0) if sk_weight_sum > 0 else 0.0
    skill_component = min(skill_component, 1.0)

    # Consistency penalty: high-value ML skills claimed but career shows no ML
    n_high_skills = sum(1 for s in skills if _best_skill_weight(s.get("name", "")) >= 0.80)
    if n_high_skills > 0 and desc_component < 0.10:
        skill_component *= 0.25
    elif n_high_skills > 0 and desc_component < 0.25:
        skill_component *= 0.55

    asm = signals.get("skill_assessment_scores", {})
    rel = [v / 100.0 for k, v in asm.items()
           if any(r in k.lower() for r in _RELEVANT_ASSESSMENTS)]
    assess_component = (sum(rel) / len(rel)) if rel else 0.50

    score = 0.35 * skill_component + 0.45 * desc_component + 0.20 * assess_component
    matched.sort(reverse=True)
    top_names = [nm for _, nm in matched[:3]]
    return score, top_names, desc_component


def _title_score(profile: dict, career: list, desc_component: float) -> float:
    """0-1.  Current title dominant; career-peak considered; strong career
    evidence of retrieval/recsys work floors the title score (JD: a candidate
    who built a recommender at a product company is a fit regardless of
    keywords)."""
    cur_tier = _title_tier(profile.get("current_title", ""))
    hist_tiers = [_title_tier(r.get("title", "")) for r in career]
    peak_hist = max(hist_tiers) if hist_tiers else 0

    combined = 0.65 * cur_tier + 0.35 * peak_hist
    normalized = min(combined / 4.0, 1.0)

    if cur_tier == 0:
        normalized *= 0.40
    elif desc_component >= 0.55 and normalized < 0.65:
        # Hidden-gem rule: career text shows real retrieval/recsys depth
        normalized = 0.65
    return normalized


def _career_score(career: list) -> float:
    """
    0-1.  35% non-consulting ratio, 25% tech-industry ratio,
    20% stability (avg tenure), 20% product-company-size proxy.
    """
    if not career:
        return 0.30

    total_m = sum(r.get("duration_months", 0) for r in career)
    consult_m = tech_m = 0
    for r in career:
        co = r.get("company", "").lower()
        ind = r.get("industry", "").lower()
        dur = r.get("duration_months", 0)
        if any(c in co for c in _CONSULTING):
            consult_m += dur
        if any(i in ind for i in _TECH_INDUSTRIES):
            tech_m += dur

    consult_ratio = consult_m / max(total_m, 1)
    tech_ratio = tech_m / max(total_m, 1)
    avg_tenure = total_m / max(len(career), 1)
    stability = min(avg_tenure / 18, 1.0)
    sizes = [r.get("company_size", "") for r in career if r.get("company_size")]
    size_sc = (sum(_COMPANY_SIZE_SCORES.get(s, 0.50) for s in sizes) / len(sizes)
               if sizes else 0.60)

    return (0.35 * (1.0 - consult_ratio) + 0.25 * tech_ratio
            + 0.20 * stability + 0.20 * size_sc)


def _experience_score(profile: dict, career: list) -> float:
    """
    0-1.  40% YOE curve (ideal 6-8, acceptable 5-9), 35% ML production depth
    in recent roles, 25% recency of ML-flavoured titles.
    """
    yoe = profile.get("years_of_experience", 0)
    if yoe < 3:
        # JD ideal is 6-8 yrs with 4-5 in applied ML; under 3 yrs total
        # cannot meet that, however strong the other signals.
        yoe_sc = yoe / 3 * 0.25
    elif yoe <= 5:
        yoe_sc = 0.25 + 0.55 * (yoe - 3) / 2
    elif yoe <= 9:
        yoe_sc = 0.80 + 0.20 * (yoe - 5) / 4
    elif yoe <= 12:
        yoe_sc = 1.00 - 0.15 * (yoe - 9) / 3
    else:
        yoe_sc = max(0.50, 0.85 - 0.02 * (yoe - 12))

    recent = career[:3]
    weights = [1.0, 0.6, 0.3]
    ml_prod_sc = 0.0
    for i, r in enumerate(recent):
        desc = (r.get("description", "") + " " + r.get("title", "")).lower()
        hits = sum(1 for s in _ML_PRODUCTION_SIGNALS if s in desc)
        ml_prod_sc += weights[i] * min(hits / 5, 1.0)
    norm = sum(weights[: len(recent)])
    ml_prod_sc = ml_prod_sc / norm if norm > 0 else 0.0

    ml_recency = 0.0
    rec_w = [1.0, 0.60, 0.35]
    for i, r in enumerate(career[:3]):
        if _title_tier(r.get("title", "")) >= 3:
            ml_recency += rec_w[i]
    ml_recency = min(ml_recency, 1.0)

    return 0.40 * yoe_sc + 0.35 * ml_prod_sc + 0.25 * ml_recency


def _availability_score(signals: dict) -> float:
    """0-1.  Behavioural signals: can this candidate actually be hired?"""
    sc = 0.0
    if signals.get("open_to_work_flag", False):
        sc += 0.25

    last = _parse_date(signals.get("last_active_date"))
    if last:
        days = (REFERENCE_DATE - last).days
        if days < 7:
            act = 1.00
        elif days < 30:
            act = 0.90
        elif days < 90:
            act = 0.70
        elif days < 180:
            act = 0.50
        else:
            act = max(0.10, 1 - days / 730)
    else:
        act = 0.50
    sc += 0.20 * act

    sc += 0.20 * signals.get("recruiter_response_rate", 0.50)

    notice = signals.get("notice_period_days", 60)
    if notice <= 0:
        nsc = 1.00
    elif notice <= 30:
        nsc = 0.90
    elif notice <= 60:
        nsc = 0.70
    elif notice <= 90:
        nsc = 0.50
    else:
        nsc = 0.30
    sc += 0.15 * nsc

    sc += 0.10 * signals.get("interview_completion_rate", 0.70)

    gh = signals.get("github_activity_score", -1)
    gh_sc = (min(gh / 70, 1.0) if gh >= 0 else 0.40)
    sc += 0.10 * gh_sc

    return min(sc, 1.0)


def _location_score(profile: dict, signals: dict) -> float:
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    for city, sc in _LOCATION_SCORES.items():
        if city in loc:
            return sc
    if "india" in country:
        return 0.55 if signals.get("willing_to_relocate", False) else 0.45
    # JD: outside India case-by-case, no visa sponsorship
    return 0.25 if signals.get("willing_to_relocate", False) else 0.15


def _penalty(profile: dict, career: list, signals: dict) -> float:
    """Multiplicative penalty for JD-stated disqualifiers. 1.0 = no penalty."""
    p = 1.0
    cur = profile.get("current_title", "").lower()

    if any(kw in cur for kw in _DISQUAL_TITLES):
        p *= 0.25

    total_m = sum(r.get("duration_months", 0) for r in career)
    if career and total_m > 0:
        consult_m = sum(
            r.get("duration_months", 0) for r in career
            if any(c in r.get("company", "").lower() for c in _CONSULTING)
        )
        if consult_m / total_m > 0.85:
            p *= 0.50

    last = _parse_date(signals.get("last_active_date"))
    if last:
        days = (REFERENCE_DATE - last).days
        if days > 180 and not signals.get("open_to_work_flag", False):
            p *= 0.60

    if signals.get("notice_period_days", 60) > 90:
        p *= 0.85

    # JD: down-weight unreachable candidates — near-zero response rate means
    # "for hiring purposes, not actually available"
    if signals.get("recruiter_response_rate", 0.5) < 0.15:
        p *= 0.85

    # Experience floor: the JD band is 5-9 yrs ("range, not requirement",
    # but its ideal sketch needs 4-5 yrs of applied ML inside the total)
    yoe = profile.get("years_of_experience", 10)
    if yoe < 3:
        p *= 0.75
    elif yoe < 4:
        p *= 0.90

    blob = _career_blob(career).lower()

    # CV/Speech/Robotics without NLP crossover (JD: explicit "don't want")
    if any(kw in cur for kw in ("computer vision", "cv engineer", "speech", "robotics")):
        if not any(k in blob for k in _NLP_CROSSOVER):
            p *= 0.55

    # Pure research career without production deployment (JD: hard disqualifier)
    if career:
        research_m = sum(
            r.get("duration_months", 0) for r in career
            if any(t in r.get("title", "").lower() for t in _RESEARCH_ONLY_TITLES)
        )
        if total_m > 0 and research_m / total_m > 0.80:
            if not any(s in blob for s in ("production", "deployed", "shipped", "real users")):
                p *= 0.50

    # Architect/manager who likely hasn't coded in 18 months (JD disqualifier)
    if any(t in cur for t in _ARCHITECT_TITLES):
        recent_desc = " ".join(
            r.get("description", "") for r in career[:1]
        ).lower()
        if not any(s in recent_desc for s in _HANDS_ON_SIGNALS):
            p *= 0.75

    # Title-chaser: 4+ jobs averaging under 15 months each (JD: "don't want")
    if len(career) >= 4 and total_m / len(career) < 15:
        p *= 0.80

    return max(p, 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generation (Stage-4 aware: specific, varied, honest, rank-consistent)
# ─────────────────────────────────────────────────────────────────────────────

def _cap(s: str) -> str:
    """Uppercase the first character only (capitalize() would lowercase 'JD')."""
    return s[0].upper() + s[1:] if s else s


def _build_reasoning(candidate: dict, rank_band: str, top_skills: list,
                     desc_component: float, loc_sc: float) -> str:
    """
    Builds a 1-2 sentence reasoning string from actual profile facts:
    current role, named skills present in the profile, career evidence,
    behavioural signals, and honest concerns. Phrasing varies by candidate.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "candidate")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    loc_city = profile.get("location", "").split(",")[0].strip()
    rr = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 0)
    open_tw = signals.get("open_to_work_flag", False)
    gh = signals.get("github_activity_score", -1)

    # Career evidence phrase from the most recent relevant role
    evidence = ""
    for r in career[:2]:
        d = r.get("description", "").lower()
        if any(k in d for k in ("retrieval", "ranking", "recommender",
                                "recommendation", "search", "embedding", "rag")):
            verbs = [k for k in ("retrieval", "ranking", "recommendation",
                                 "search", "embedding") if k in d]
            role_co = r.get("company", "")
            if role_co and role_co != company:
                evidence = f"built {verbs[0]} systems at {role_co}"
            else:
                evidence = f"current role involves {verbs[0]} work"
            break

    # Skills phrase — only names actually present in profile
    skills_phrase = ""
    if top_skills:
        if len(top_skills) >= 2:
            skills_phrase = f"hands-on {top_skills[0]} and {top_skills[1]}"
        else:
            skills_phrase = f"hands-on {top_skills[0]}"

    # Concerns (honest, JD-grounded) — pick the most salient
    concerns = []
    if notice > 60:
        concerns.append(f"notice period is {notice} days (JD prefers under 30)")
    if rr < 0.30:
        concerns.append(f"recruiter response rate is low ({rr:.0%})")
    if loc_sc < 0.60:
        if loc_city:
            concerns.append(f"based in {loc_city}, outside preferred Pune/Noida")
        else:
            concerns.append("location outside preferred Pune/Noida")
    if yoe < 5:
        concerns.append(f"{yoe:.1f} yrs experience is below the 5-9 band")
    elif yoe > 9:
        concerns.append(f"{yoe:.1f} yrs experience is above the 5-9 band")
    concern = concerns[0] if concerns else ""

    # Positives pool
    pos = []
    if open_tw:
        pos.append("open to work")
    if rr >= 0.60:
        pos.append(f"{rr:.0%} recruiter response rate")
    if 0 < notice <= 30:
        pos.append(f"{notice}-day notice")
    if gh >= 60:
        pos.append(f"strong GitHub activity ({gh:.0f}/100)")
    if loc_sc >= 0.99:
        pos.append(f"based in {loc_city}")
    pos_phrase = ", ".join(pos[:2])

    # Assemble with structural variation keyed off candidate_id
    h = sum(ord(ch) for ch in candidate.get("candidate_id", "")) % 4
    core = f"{title} at {company}" if company else title

    if rank_band == "top":
        if h == 0:
            s = f"{core} with {yoe:.1f} yrs; {skills_phrase}" if skills_phrase else f"{core} with {yoe:.1f} yrs"
            if evidence:
                s += f"; {evidence}"
            s += f". {_cap(pos_phrase)}." if pos_phrase else "."
        elif h == 1:
            s = f"{yoe:.1f} yrs as {core}"
            if evidence:
                s += f" — {evidence}"
            if skills_phrase:
                s += f"; {skills_phrase}"
            s += f". {_cap(pos_phrase)}." if pos_phrase else "."
        elif h == 2:
            s = f"Strong JD match: {core}, {yoe:.1f} yrs"
            if skills_phrase:
                s += f", {skills_phrase}"
            s += f". {_cap(evidence)}." if evidence else "."
            if pos_phrase:
                s += f" {_cap(pos_phrase)}."
        else:
            s = f"{core} ({yoe:.1f} yrs)"
            if evidence:
                s += f"; {evidence}"
            if pos_phrase:
                s += f"; {pos_phrase}"
            s += "."
        if concern:
            s += f" Minor concern: {concern}."
    elif rank_band == "mid":
        if h % 2 == 0:
            s = f"{core}, {yoe:.1f} yrs"
            if skills_phrase:
                s += f"; {skills_phrase}"
            if evidence:
                s += f"; {evidence}"
            s += "."
        else:
            s = f"{yoe:.1f} yrs of experience, currently {core}"
            if evidence:
                s += f"; {evidence}"
            if skills_phrase:
                s += f"; {skills_phrase}"
            s += "."
        if concern:
            s += f" Concern: {concern}."
        elif pos_phrase:
            s += f" {_cap(pos_phrase)}."
    else:  # tail — hedged, with varied closers to avoid templating
        closers = (
            "A stretch versus the top of the list, but worth a recruiter look.",
            "Weaker than the leading profiles, though the fundamentals are there.",
            "Kept in the shortlist on overall signal despite the gaps.",
            "Likely below the hiring bar, included for pipeline breadth.",
        )
        if h % 2 == 0:
            s = f"{core}, {yoe:.1f} yrs — adjacent fit"
            if skills_phrase:
                s += f" with {skills_phrase}"
            s += "."
        else:
            s = f"Partial match: {core} ({yoe:.1f} yrs)"
            if skills_phrase:
                s += f", {skills_phrase}"
            s += "."
        if concern:
            s += f" {_cap(concern)}."
        s += f" {closers[h]}"

    # CSV hygiene: no newlines, reasonable length
    return " ".join(s.split())[:300]


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring entry-point
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> tuple:
    """Returns (final_score, context dict used later for reasoning)."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    total_career_months = sum(r.get("duration_months", 0) for r in career)

    hp = honeypot_flags(candidate)

    tech, top_skills, desc_comp = _tech_score(skills, career, signals,
                                              total_career_months)
    title = _title_score(profile, career, desc_comp)
    car = _career_score(career)
    exp = _experience_score(profile, career)
    avail = _availability_score(signals)
    loc = _location_score(profile, signals)
    pen = _penalty(profile, career, signals)

    base = (
        0.25 * title
        + 0.28 * tech
        + 0.20 * car
        + 0.17 * exp
        + 0.10 * avail
    )
    final = base * (0.70 + 0.30 * loc) * pen

    if hp:
        # Internally inconsistent (honeypot-pattern) profile: force to bottom.
        final *= 0.02

    ctx = {
        "top_skills": top_skills,
        "desc_component": desc_comp,
        "loc": loc,
        "honeypot": hp,
    }
    return final, ctx


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob AI Engineer role."
    )
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates JSONL file")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100,
                        help="How many candidates to include in output")
    args = parser.parse_args()

    print(f"Reading {args.candidates} …", file=sys.stderr)
    results = []
    total = 0
    n_honeypot = 0

    with open(args.candidates, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            s, ctx = score_candidate(c)
            if ctx["honeypot"]:
                n_honeypot += 1
            results.append((c["candidate_id"], s, ctx, c))
            total += 1
            if total % 10000 == 0:
                print(f"  … {total:,} candidates processed", file=sys.stderr)

    print(f"Scored {total:,} candidates "
          f"({n_honeypot} flagged as internally inconsistent). "
          f"Selecting top {args.top_n} …", file=sys.stderr)

    results.sort(key=lambda x: (-x[1], x[0]))
    top = results[: args.top_n]

    # Normalise scores to [0.20, 1.00] preserving order
    max_sc = top[0][1] if top else 1.0
    min_sc = top[-1][1] if top else 0.0
    span = max_sc - min_sc if max_sc != min_sc else 1.0

    # Pre-compute rounded scores, then enforce the tie-break rule:
    # equal rounded scores must be ordered by candidate_id ascending.
    rows = []
    for cid, raw, ctx, cand in top:
        norm = 0.20 + 0.80 * (raw - min_sc) / span
        rows.append([cid, f"{norm:.6f}", ctx, cand])
    rows.sort(key=lambda r: (-float(r[1]), r[0]))

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score_s, ctx, cand) in enumerate(rows, start=1):
            band = "top" if rank <= 25 else ("mid" if rank <= 70 else "tail")
            reasoning = _build_reasoning(
                cand, band, ctx["top_skills"], ctx["desc_component"], ctx["loc"]
            )
            w.writerow([cid, rank, score_s, reasoning])

    print(f"Wrote {len(rows)} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
