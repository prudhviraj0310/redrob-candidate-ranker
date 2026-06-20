# Redrob AI Candidate Discovery & Ranking Challenge — Team Prudhvi

This repository contains the intelligent candidate ranking system developed for the Redrob AI hackathon.

## Challenge Goal
The objective is to discover and rank the top 100 candidate profiles from a pool of 100,000 candidates (`candidates.jsonl`) for a **Senior AI Engineer (Founding Team)** role at Redrob AI, satisfying all requirements of the job description and avoiding synthetic traps (honeypots, keyword stuffers, consulting-only profiles).

## Setup & Reproduction

To reproduce the submission CSV file under 20 seconds, run:

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Dependencies
No external packages are required (runs on Python 3 standard library). This guarantees high performance, zero network footprint, and 100% reliability in CPU-only sandboxed evaluation environments.

## Methodology & Architecture

Our candidate scoring model is built on five weighted components:

1. **Role Fit (35%):** Uses structural title tiering and checks candidate career descriptions for production search, recommendation, and retrieval system experience. This acts as a "hidden gem" detector for candidates with generic titles but highly relevant experience, while downweighting keyword-stuffer profiles.
2. **Skills Match (25%):** A tiered skills parser prioritizing retrieval techniques, vector databases, machine learning, and ranking evaluation metrics (NDCG, MRR, MAP). Proficiency, duration, and endorsement counts act as modifiers.
3. **Experience Quality (15%):** Prioritizes the 5-9 years sweet spot, penalizes pure research environments without production deployments, and checks the ratio of ML roles in candidate histories.
4. **Location (10%):** Prioritizes Noida/Pune (preferred), Tier-1 Indian cities, and other India-based candidates.
5. **Behavioral Signals (15%):** Actively checks availability and activity recency, recruiter response rate, notice period (preferring <30 days), and interview completion rate.

### Filter & Penalty Layers
- **Honeypot Filter:** Detects impossible profiles (e.g., 3+ expert skills with 0 months duration, career date inconsistencies where claimed job duration exceeds calendar span by more than 6 months, and single job duration exceeding total years of experience).
- **Consulting Filter:** Filters out candidates whose entire career has been spent at consulting companies.
- **Title-Chaser Penalty:** Penalizes candidates with a history of quick job hopping (average tenure < 18 months).
- **Domain Mismatch Penalty:** Penalizes CV/Speech/Robotics-only candidates with zero NLP/IR exposure.

### Reasoning Generation (Stage 4 Compliance)
A custom reasoning generator builds a unique 1-2 sentence justification for every ranked candidate. It utilizes candidate-specific facts (specific companies, years of experience, direct skill profiles) and links them to the job description, while highlighting honest concerns (long notice period, location relocation, experience range mismatches). It avoids templates to prevent Stage 4 manual review penalties.
