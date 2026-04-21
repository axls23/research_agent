# PRISMA Stage Mapping Decision

This document defines the canonical mapping from pipeline state to PRISMA counters.

## Why this exists

The same run was previously interpreted in multiple ways by different nodes, which produced impossible metrics (for example, included > screened). This mapping is the source of truth for audit export and flow diagrams.

## Canonical mapping

- identified_records: `papers_found`
- screened_records: `max(papers_screened, len(papers))`, then clamped to `identified_records`
- full_text_assessed: count of papers where `paper.full_text` is present
- studies_included: count of papers where `paper.included == true` and `paper.full_text` is present

## Invariants

These MUST always hold in exported artifacts:

- `identified_records >= screened_records`
- `screened_records >= full_text_assessed`
- `full_text_assessed >= studies_included`
- `excluded_at_screening = identified_records - screened_records`
- `excluded_at_full_text = full_text_assessed - studies_included`

## Notes for maintainers

- `papers_included` in mutable state may still be written by intermediate nodes for legacy compatibility.
- Audit export and PRISMA flow visualization should always rely on computed canonical counters rather than trusting a single mutable field.
