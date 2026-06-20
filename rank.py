"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking
=============================================================
Ranks 100K candidates against a Senior AI Engineer (Founding Team) JD.

Architecture:
  1. Load all candidates from JSONL
  2. Filter honeypots (composite score: career impossibilities, keyword stuffers)
  3. Apply JD-specified disqualifiers (consulting-only, title-chasers, pure research, etc.)
  4. Score via 5 weighted components: role_fit, skills, experience, location, behavioral
  5. Generate unique, JD-connected reasoning per candidate
  6. Output top-100 CSV

Constraints: <=5 min, <=16 GB RAM, CPU only, no network.
"""

import json
import csv
import sys
import argparse
import math
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOW = datetime(2026, 6, 20)

# Core skills the JD explicitly requires or values
CORE_RETRIEVAL_SKILLS = {
    'embeddings', 'sentence-transformers', 'sentence transformers',
    'vector database', 'vector search', 'hybrid search',
    'semantic search', 'information retrieval', 'search infrastructure',
    'search backend', 'ranking', 'ranking systems', 'learning to rank',
    're-ranking', 'reranking', 'rag', 'retrieval augmented generation',
    'bge', 'e5',
}

CORE_VECTORDB_SKILLS = {
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',
    'opensearch', 'elasticsearch', 'pgvector', 'chromadb',
}

CORE_ML_SKILLS = {
    'python', 'pytorch', 'tensorflow', 'llm', 'llms',
    'fine-tuning', 'fine-tuning llms', 'finetuning', 'lora', 'qlora', 'peft',
    'nlp', 'natural language processing', 'transformers', 'huggingface',
    'xgboost', 'lightgbm',
}

SUPPORTING_SKILLS = {
    'spark', 'airflow', 'data pipelines', 'sql', 'git', 'docker',
    'kubernetes', 'aws', 'gcp', 'azure', 'cloud', 'mlflow',
    'weights & biases', 'wandb', 'dvc',
}

# Evaluation metric keywords (JD explicitly values eval expertise)
EVAL_KEYWORDS = {'ndcg', 'mrr', 'map', 'precision', 'recall', 'f1',
                 'a/b test', 'ab test', 'evaluation', 'benchmark'}

# Non-AI titles — candidates with these titles AND many AI skills = keyword stuffer
NON_AI_TITLES = {
    'marketing manager', 'hr manager', 'human resources',
    'accountant', 'accounting', 'sales executive', 'sales manager',
    'civil engineer', 'mechanical engineer', 'graphic designer',
    'content writer', 'customer support', 'operations manager',
    'project manager', 'business analyst', 'financial analyst',
    'teacher', 'professor', 'lecturer', 'admin',
}

# Consulting firms (JD explicit disqualifier for consulting-only careers)
CONSULTING_FIRMS = {
    'tcs', 'tata consultancy services', 'tata consultancy',
    'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'hcl', 'hcl technologies', 'tech mahindra', 'l&t infotech',
    'lnt infotech', 'l&t technology services', 'mindtree',
    'mphasis', 'hexaware', 'cyient', 'zensar', 'persistent systems',
}

# Production-indicating keywords in career descriptions
PRODUCTION_KEYWORDS = [
    'production', 'deployed', 'shipped', 'launched', 'live',
    'real users', 'real-time', 'scale', 'at scale',
    'millions', 'thousands of users', 'user-facing',
    'a/b test', 'online experiment', 'latency', 'throughput',
    'api', 'microservice', 'pipeline', 'system design',
]

# Retrieval/search/ranking career keywords (for "hidden gem" detection)
RETRIEVAL_CAREER_KEYWORDS = [
    'recommendation system', 'recommendation engine', 'recommender',
    'collaborative filtering', 'matrix factorization',
    'search engine', 'information retrieval', 'hybrid search',
    'vector search', 'vector database', 'neural search',
    'semantic search', 'rag', 'retrieval augmented',
    'embeddings', 'embedding', 'ranking system', 'learning to rank',
    'candidate retrieval', 'query understanding', 'relevance',
    'bm25', 'tf-idf', 'inverted index',
]

# CV/Speech/Robotics keywords (JD explicitly doesn't want these without NLP/IR)
CV_SPEECH_ROBOTICS = {
    'computer vision', 'image classification', 'object detection',
    'image segmentation', 'yolo', 'cnn', 'convolutional',
    'speech recognition', 'speech synthesis', 'tts', 'asr',
    'robotics', 'ros', 'autonomous', 'lidar', 'slam',
    'gan', 'gans', 'generative adversarial',
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def parse_date(date_str):
    """Parse a YYYY-MM-DD date string; return NOW on failure."""
    if not date_str:
        return NOW
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return NOW


def skill_matches(skill_name_lower, keyword_set):
    """Check if a skill name (lowered) matches any keyword in the set."""
    for kw in keyword_set:
        if kw in skill_name_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. HONEYPOT DETECTION
# ---------------------------------------------------------------------------
def compute_honeypot_score(c):
    """
    Compute a composite honeypot score. Higher = more suspicious.
    The dataset has ~80 honeypots with "subtly impossible profiles":
      - "8 years experience at company founded 3 years ago"
      - "expert proficiency in 10 skills with 0 years used"
    
    IMPORTANT: Keyword stuffers (HR Manager with AI skills) are NOT honeypots.
    They are simply bad candidates scored low via role_fit. Honeypots are
    specifically candidates with IMPOSSIBLE profile data.
    """
    signals = 0
    skills = c.get("skills", [])
    history = c.get("career_history", [])
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)

    # Signal 1: Multiple expert skills with 0 duration_months
    # Problem statement: "expert proficiency in 10 skills with 0 years used"
    # Data shows candidates with 3-5 expert skills at 0 duration
    expert_0 = [s for s in skills if
                s.get("proficiency") == "expert" and
                s.get("duration_months", 1) == 0]
    if len(expert_0) >= 3:
        signals += 2  # Strong signal — impossible to be expert with 0 usage

    # Signal 2: Career date impossibility
    # "8 years experience at company founded 3 years ago"
    # duration_months significantly exceeds (end_date - start_date)
    for job in history:
        start_str = job.get("start_date")
        duration = job.get("duration_months", 0)
        if not start_str or duration is None:
            continue
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            end_str = job.get("end_date")
            end_dt = datetime.strptime(end_str, "%Y-%m-%d") if end_str else NOW
            actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            # If claimed duration exceeds actual calendar span by >6 months
            if duration > actual_months + 6:
                signals += 2
                break
        except (ValueError, TypeError):
            pass

    # Signal 3: Single job duration exceeds total years of experience
    # This is a data impossibility — you can't work 10 years at one company
    # if you only have 5 years total experience
    for job in history:
        duration = job.get("duration_months", 0)
        if duration and yoe and yoe > 0 and (duration / 12.0) > yoe + 1.0:
            signals += 2
            break

    return signals


def is_honeypot(c):
    """A candidate is flagged as honeypot if composite score >= 2."""
    return compute_honeypot_score(c) >= 2.0


# ---------------------------------------------------------------------------
# 2. JD-SPECIFIED DISQUALIFIERS
# ---------------------------------------------------------------------------
def is_consulting_only(c):
    """
    JD: "People who have only worked at consulting firms in their entire career"
    BUT: "if you're currently at one but have prior product-company experience, fine"
    """
    history = c.get("career_history", [])
    if not history:
        return False
    companies = [job.get("company", "").strip().lower() for job in history if job.get("company")]
    if not companies:
        return False
    # Check if ALL companies are consulting firms
    all_consulting = all(
        any(cf in comp for cf in CONSULTING_FIRMS)
        for comp in companies
    )
    return all_consulting


def get_title_chaser_penalty(c):
    """
    JD: "Title-chasers switching companies every 1.5 years optimizing for titles"
    Penalize if avg tenure < 18 months across 3+ jobs.
    """
    history = c.get("career_history", [])
    if len(history) < 3:
        return 0.0
    durations = [job.get("duration_months", 0) for job in history if not job.get("is_current", False)]
    if len(durations) < 3:
        return 0.0
    avg_tenure = sum(durations) / len(durations)
    if avg_tenure < 14:
        return 0.4  # Heavy penalty
    elif avg_tenure < 18:
        return 0.2  # Moderate penalty
    return 0.0


def get_cv_speech_only_penalty(c):
    """
    JD: "People whose primary expertise is computer vision, speech, or robotics
    without significant NLP/IR exposure."
    """
    skills = c.get("skills", [])
    cv_speech_count = 0
    nlp_ir_count = 0
    for s in skills:
        name = s.get("name", "").lower()
        if skill_matches(name, CV_SPEECH_ROBOTICS):
            cv_speech_count += 1
        if skill_matches(name, CORE_RETRIEVAL_SKILLS | {'nlp', 'natural language processing'}):
            nlp_ir_count += 1
    if cv_speech_count >= 3 and nlp_ir_count == 0:
        return 0.5  # Significant penalty
    return 0.0


def has_production_experience(c):
    """Check if candidate has any production deployment experience."""
    for job in c.get("career_history", []):
        desc = job.get("description", "").lower()
        if any(kw in desc for kw in PRODUCTION_KEYWORDS):
            return True
    return False


def get_pure_research_penalty(c):
    """
    JD: "If career is pure research environments without any production deployment"
    """
    history = c.get("career_history", [])
    title = c.get("profile", {}).get("current_title", "").lower()

    research_indicators = ['research scientist', 'researcher', 'research fellow',
                           'postdoc', 'phd student', 'research assistant',
                           'research associate']
    is_research_title = any(ri in title for ri in research_indicators)

    if is_research_title and not has_production_experience(c):
        return 0.4
    return 0.0


# ---------------------------------------------------------------------------
# 3. SCORING COMPONENTS
# ---------------------------------------------------------------------------

def get_role_fit_score(c):
    """
    Score role relevance based on title + career history.
    JD: "The right answer is not 'find candidates whose skills contain the most AI keywords.'
    A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' but if their career history
    shows they built a recommendation system, they're a fit."
    """
    profile = c.get("profile", {})
    title = profile.get("current_title", "").lower()
    summary = profile.get("summary", "").lower()
    industry = profile.get("current_industry", "").lower()
    history = c.get("career_history", [])

    # Tier 1 titles (strong match)
    tier1_titles = ['search engineer', 'ranking engineer', 'recommendation',
                    'retrieval engineer', 'nlp engineer', 'applied scientist',
                    'applied ml', 'ai engineer', 'ml engineer',
                    'machine learning engineer', 'senior ai engineer',
                    'staff machine learning', 'lead ai', 'lead ml']

    # Tier 2 titles (good match)
    tier2_titles = ['data scientist', 'ai specialist', 'ai research',
                    'software engineer (ml)', 'senior software engineer (ml)',
                    'deep learning', 'analytics engineer']

    # Tier 3 titles (possible match — need career evidence)
    tier3_titles = ['software engineer', 'backend engineer', 'data engineer',
                    'platform engineer', 'full stack']

    # Calculate title score
    title_score = 0.1  # default for unknown
    for t1 in tier1_titles:
        if t1 in title:
            title_score = 1.0
            break
    if title_score < 1.0:
        for t2 in tier2_titles:
            if t2 in title:
                title_score = 0.75
                break
    if title_score < 0.75:
        for t3 in tier3_titles:
            if t3 in title:
                title_score = 0.45
                break

    # Non-AI title penalty
    is_non_ai = any(nt in title for nt in NON_AI_TITLES)
    if is_non_ai:
        title_score = 0.02

    # Career history: look for retrieval/search/ranking production experience
    career_bonus = 0.0
    has_retrieval_production = False
    product_company_count = 0

    for job in history:
        desc = job.get("description", "").lower()
        job_title = job.get("title", "").lower()
        job_industry = job.get("industry", "").lower()

        # Check for retrieval/search/ranking work
        if any(kw in desc for kw in RETRIEVAL_CAREER_KEYWORDS):
            career_bonus += 0.15
            has_retrieval_production = True

        # Production deployment experience
        if any(kw in desc for kw in PRODUCTION_KEYWORDS):
            career_bonus += 0.05

        # Product company (not consulting)
        company = job.get("company", "").lower()
        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        if not is_consulting and job.get("company_size", "") not in ["", "1-10"]:
            product_company_count += 1

    # Cap career bonus
    career_bonus = min(career_bonus, 0.3)

    # Product company experience bonus
    if product_company_count >= 2:
        career_bonus += 0.1

    # "Hidden gem" boost: if career shows retrieval work but title is generic
    if has_retrieval_production and title_score < 0.75:
        title_score = max(title_score, 0.6)

    # Summary analysis — look for relevant experience mentions
    summary_bonus = 0.0
    summary_keywords = ['retrieval', 'search', 'ranking', 'recommendation',
                        'embeddings', 'vector', 'nlp', 'llm', 'fine-tun',
                        'information retrieval', 'rag']
    summary_matches = sum(1 for kw in summary_keywords if kw in summary)
    summary_bonus = min(summary_matches * 0.03, 0.1)

    return min(title_score + career_bonus + summary_bonus, 1.0)


def get_skills_score(c):
    """
    Score skills relevance with tiered weighting.
    JD explicitly requires: embeddings-based retrieval, vector databases, Python,
    evaluation frameworks (NDCG/MRR/MAP).
    """
    skills = c.get("skills", [])

    proficiency_mult = {
        'expert': 1.6,
        'advanced': 1.3,
        'intermediate': 1.0,
        'beginner': 0.5,
    }

    total_weighted = 0.0
    has_retrieval = False
    has_vectordb = False
    has_python = False
    has_eval = False

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "intermediate")
        dur = s.get("duration_months", 0)
        endorse = s.get("endorsements", 0)

        # Determine skill tier
        weight = 0.0
        if skill_matches(name, CORE_RETRIEVAL_SKILLS):
            weight = 2.0
            has_retrieval = True
        elif skill_matches(name, CORE_VECTORDB_SKILLS):
            weight = 1.8
            has_vectordb = True
        elif skill_matches(name, CORE_ML_SKILLS):
            weight = 1.2
            if 'python' in name:
                has_python = True
        elif skill_matches(name, EVAL_KEYWORDS):
            weight = 1.5
            has_eval = True
        elif skill_matches(name, SUPPORTING_SKILLS):
            weight = 0.4

        if weight > 0:
            p_mult = proficiency_mult.get(prof, 1.0)
            # Duration factor: diminishing returns after 48 months
            dur_factor = min(dur, 60) / 24.0 if dur > 0 else 0.1
            # Endorsement factor: log scale
            endorse_factor = 1.0 + math.log1p(endorse) * 0.08
            total_weighted += weight * p_mult * dur_factor * endorse_factor

    # Normalize
    raw_score = min(total_weighted / 20.0, 1.0)

    # Bonus for having the JD "must-have" skills
    must_have_bonus = 0.0
    if has_retrieval:
        must_have_bonus += 0.08
    if has_vectordb:
        must_have_bonus += 0.05
    if has_python:
        must_have_bonus += 0.03
    if has_eval:
        must_have_bonus += 0.04

    return min(raw_score + must_have_bonus, 1.0)


def get_experience_score(c):
    """
    Score experience quality.
    JD: "5-9 years" ideal; "4-5 are in applied ML/AI roles at product companies"
    """
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    history = c.get("career_history", [])

    # Years of experience fit
    if 5.0 <= yoe <= 9.0:
        yoe_score = 1.0
    elif 4.0 <= yoe < 5.0:
        yoe_score = 0.8
    elif 9.0 < yoe <= 11.0:
        yoe_score = 0.75
    elif 3.0 <= yoe < 4.0:
        yoe_score = 0.5
    elif 11.0 < yoe <= 14.0:
        yoe_score = 0.4
    elif yoe > 14.0:
        yoe_score = 0.25
    else:
        yoe_score = 0.15

    # Career quality: product companies, AI/tech industries
    career_quality = 0.0
    ml_role_months = 0
    total_months = 0
    for job in history:
        dur = job.get("duration_months", 0)
        total_months += dur
        job_title = job.get("title", "").lower()
        industry = job.get("industry", "").lower()

        # AI/ML role months
        ml_indicators = ['ml', 'ai', 'machine learning', 'data scien',
                         'nlp', 'search', 'ranking', 'recommendation',
                         'applied scientist']
        if any(ind in job_title for ind in ml_indicators):
            ml_role_months += dur

        # Product tech company bonus
        tech_industries = ['ai', 'ml', 'technology', 'software', 'internet',
                           'e-commerce', 'fintech', 'saas', 'cloud',
                           'platform', 'marketplace']
        if any(ti in industry for ti in tech_industries):
            career_quality += 0.05

    career_quality = min(career_quality, 0.2)

    # Ratio of ML experience to total
    if total_months > 0:
        ml_ratio = ml_role_months / total_months
        ml_ratio_score = min(ml_ratio, 1.0) * 0.2
    else:
        ml_ratio_score = 0.0

    return min(yoe_score * 0.6 + career_quality + ml_ratio_score + 0.2, 1.0)


def get_location_score(c):
    """
    JD: "Pune/Noida preferred. Open to Tier-1 Indian cities.
    Outside India: case-by-case, no visa sponsorship."
    """
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing = signals.get("willing_to_relocate", False)

    preferred = {'pune', 'noida'}
    tier1 = {'delhi', 'ncr', 'gurgaon', 'gurugram', 'ghaziabad', 'faridabad',
             'mumbai', 'navi mumbai', 'hyderabad', 'bangalore', 'bengaluru',
             'chennai', 'kolkata'}

    is_india = 'india' in country or 'india' in loc

    if is_india:
        if any(city in loc for city in preferred):
            return 1.0
        elif any(city in loc for city in tier1):
            return 0.85
        elif willing:
            return 0.7
        else:
            return 0.6
    else:
        if willing:
            return 0.3
        return 0.1


def get_behavioral_score(c):
    """
    JD: "A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is not actually available."
    Promoted from multiplier to full scoring component.
    """
    signals = c.get("redrob_signals", {})
    score = 0.0

    # 1. Recency of activity (most important behavioral signal)
    active_str = signals.get("last_active_date", "")
    active_dt = parse_date(active_str)
    days_inactive = (NOW - active_dt).days
    if days_inactive <= 14:
        score += 0.25
    elif days_inactive <= 30:
        score += 0.22
    elif days_inactive <= 60:
        score += 0.18
    elif days_inactive <= 90:
        score += 0.12
    elif days_inactive <= 180:
        score += 0.05
    else:
        score += 0.0  # Inactive > 6 months = effectively unavailable

    # 2. Open to work flag
    if signals.get("open_to_work_flag", False):
        score += 0.15
    else:
        score += 0.03

    # 3. Recruiter response rate
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    if resp_rate < 0.05:
        score += 0.0  # 5% = not responding at all
    elif resp_rate < 0.20:
        score += 0.05
    elif resp_rate < 0.50:
        score += 0.10
    else:
        score += 0.15

    # 4. Notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.10
    elif notice <= 90:
        score += 0.05
    else:
        score += 0.0  # 90+ days is painful for Series A startup

    # 5. Interview completion rate
    int_rate = signals.get("interview_completion_rate", 0.5)
    if int_rate >= 0.8:
        score += 0.10
    elif int_rate >= 0.5:
        score += 0.05
    else:
        score += 0.0

    # 6. GitHub activity (JD values open source contributions)
    gh = signals.get("github_activity_score", -1)
    if gh > 60:
        score += 0.10
    elif gh > 30:
        score += 0.05
    elif gh == -1:
        score += 0.02  # No GitHub linked — slight negative but not deal-breaker

    # 7. Profile completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 80:
        score += 0.05
    elif completeness >= 60:
        score += 0.03

    # 8. Saved by recruiters (social proof)
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        score += 0.05
    elif saved >= 2:
        score += 0.03

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# 4. COMPOSITE SCORING
# ---------------------------------------------------------------------------
def score_candidate(c):
    """
    Compute the final score for a candidate.
    Returns 0.0 for honeypots and hard disqualifiers.
    """
    # Hard filters
    if is_honeypot(c):
        return 0.0

    if is_consulting_only(c):
        return 0.0

    # Compute components
    role_fit = get_role_fit_score(c)
    skills = get_skills_score(c)
    experience = get_experience_score(c)
    location = get_location_score(c)
    behavioral = get_behavioral_score(c)

    # Weighted combination
    # Role fit and skills are most important (JD is very specific about what they need)
    raw = (role_fit * 0.35 +
           skills * 0.25 +
           experience * 0.15 +
           location * 0.10 +
           behavioral * 0.15)

    # Apply soft penalties
    penalty = 0.0
    penalty += get_title_chaser_penalty(c)
    penalty += get_cv_speech_only_penalty(c)
    penalty += get_pure_research_penalty(c)

    # Non-AI title is a near-disqualifier
    title = c.get("profile", {}).get("current_title", "").lower()
    if any(nt in title for nt in NON_AI_TITLES):
        penalty += 0.6

    final = max(raw - penalty, 0.0)
    return final


# ---------------------------------------------------------------------------
# 5. REASONING GENERATION
# ---------------------------------------------------------------------------
def generate_reasoning(c, rank, score):
    """
    Generate unique, JD-connected, honest reasoning for each candidate.
    Stage 4 checks: specific facts, JD connection, honest concerns, no hallucination,
    variation, rank consistency.
    """
    profile = c.get("profile", {})
    skills_list = c.get("skills", [])
    signals = c.get("redrob_signals", {})
    history = c.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    company_size = profile.get("current_company_size", "")
    loc = profile.get("location", "")
    country = profile.get("country", "")
    industry = profile.get("current_industry", "")

    notice = signals.get("notice_period_days", 0)
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    open_to_work = signals.get("open_to_work_flag", False)
    gh_score = signals.get("github_activity_score", -1)
    active_str = signals.get("last_active_date", "")
    active_dt = parse_date(active_str)
    days_inactive = (NOW - active_dt).days

    # Gather relevant skills
    retrieval_skills = []
    vectordb_skills = []
    ml_skills = []
    for s in skills_list:
        name = s.get("name", "")
        name_lower = name.lower()
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0)
        if skill_matches(name_lower, CORE_RETRIEVAL_SKILLS):
            retrieval_skills.append((name, prof, dur))
        elif skill_matches(name_lower, CORE_VECTORDB_SKILLS):
            vectordb_skills.append((name, prof, dur))
        elif skill_matches(name_lower, CORE_ML_SKILLS):
            ml_skills.append((name, prof, dur))

    # Find retrieval/search career experience
    retrieval_experience = []
    for job in history:
        desc = job.get("description", "").lower()
        if any(kw in desc for kw in RETRIEVAL_CAREER_KEYWORDS):
            retrieval_experience.append(f"{job.get('title', '')} at {job.get('company', '')}")

    # Build reasoning parts
    parts = []

    # Part 1: Role/experience headline (unique to candidate)
    if retrieval_experience:
        parts.append(f"Currently {title} at {company} ({industry}), with {yoe:.1f} years of experience including production work in retrieval/search ({retrieval_experience[0]})")
    elif company:
        parts.append(f"Currently {title} at {company} ({industry}) with {yoe:.1f} years of experience")
    else:
        parts.append(f"{title} with {yoe:.1f} years of experience")

    # Part 2: Skills connection to JD requirements
    skill_mentions = []
    if retrieval_skills:
        top_ret = sorted(retrieval_skills, key=lambda x: x[2], reverse=True)[:2]
        skill_mentions.append("retrieval skills: " + ", ".join(f"{s[0]} ({s[1]}, {s[2]}mo)" for s in top_ret))
    if vectordb_skills:
        top_vdb = sorted(vectordb_skills, key=lambda x: x[2], reverse=True)[:2]
        skill_mentions.append("vector DB: " + ", ".join(f"{s[0]} ({s[1]}, {s[2]}mo)" for s in top_vdb))
    if ml_skills:
        top_ml = sorted(ml_skills, key=lambda x: x[2], reverse=True)[:2]
        skill_mentions.append(", ".join(f"{s[0]} ({s[1]})" for s in top_ml))

    if skill_mentions:
        parts.append("Key match: " + "; ".join(skill_mentions[:2]))
    else:
        parts.append("Limited direct retrieval/embedding skill match but career trajectory suggests relevant systems experience")

    # Part 3: Strengths specific to JD
    strengths = []
    if 5.0 <= yoe <= 9.0:
        strengths.append("experience in the 5-9y sweet spot")
    if any(city in loc.lower() for city in ['pune', 'noida']):
        strengths.append(f"based in preferred location ({loc})")
    elif 'india' in country.lower():
        strengths.append(f"India-based ({loc})")
    if open_to_work:
        strengths.append("actively looking")
    if notice <= 30:
        strengths.append(f"available within {notice} days")
    if gh_score > 50:
        strengths.append(f"active GitHub contributor (score: {gh_score})")
    if resp_rate >= 0.7:
        strengths.append(f"high recruiter engagement ({resp_rate:.0%} response rate)")

    if strengths:
        parts.append("Strengths: " + ", ".join(strengths[:3]))

    # Part 4: Honest concerns (Stage 4 explicitly values this)
    concerns = []
    if yoe < 4.0:
        concerns.append(f"below ideal experience range ({yoe:.1f}y vs 5-9y)")
    elif yoe > 11.0:
        concerns.append(f"above ideal experience range ({yoe:.1f}y vs 5-9y)")
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")
    if resp_rate < 0.2 and resp_rate >= 0:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")
    if days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days")
    if 'india' not in country.lower():
        concerns.append(f"based outside India ({country})")
    if not retrieval_skills and not vectordb_skills:
        concerns.append("no direct retrieval/vector DB skills listed")

    if concerns:
        parts.append("Concerns: " + ", ".join(concerns[:2]))

    # Combine
    reasoning = ". ".join(parts) + "."

    # Truncate if too long (keep it to 1-2 sentences equivalent)
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."

    return reasoning


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for Senior AI Engineer (Founding Team) at Redrob AI")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates JSONL file")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    honeypot_count = 0
    consulting_count = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            c = json.loads(line)
            s = score_candidate(c)

            if is_honeypot(c):
                honeypot_count += 1
            elif is_consulting_only(c):
                consulting_count += 1

            candidates.append({
                "candidate_id": c["candidate_id"],
                "score": s,
                "record": c,
            })

            if (i + 1) % 20000 == 0:
                print(f"  Processed {i + 1} candidates...")

    print(f"Total candidates: {len(candidates)}")
    print(f"Honeypots detected: {honeypot_count}")
    print(f"Consulting-only filtered: {consulting_count}")

    # Sort by score descending, then by candidate_id ascending for tiebreaking
    # Sort by rounded score (to match CSV output precision) descending, 
    # then candidate_id ascending for tie-breaking
    candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))

    top_100 = candidates[:100]

    # Write CSV
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(top_100):
            rank = idx + 1
            reasoning = generate_reasoning(item["record"], rank, item["score"])
            writer.writerow([
                item["candidate_id"],
                rank,
                round(item["score"], 4),
                reasoning,
            ])

    print(f"\nSuccessfully wrote top 100 to {args.out}")
    print(f"  Score range: {top_100[0]['score']:.4f} — {top_100[-1]['score']:.4f}")

    # Print top 10 summary
    print("\nTop 10:")
    for i, item in enumerate(top_100[:10]):
        p = item["record"]["profile"]
        print(f"  {i+1}. {item['candidate_id']} | {p['current_title']} | "
              f"{p['years_of_experience']}y | {p['location']} | "
              f"score={item['score']:.4f}")


if __name__ == "__main__":
    main()
