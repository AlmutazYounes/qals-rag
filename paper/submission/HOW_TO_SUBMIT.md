# Submit to ESWA without paying gold OA

Target: Elsevier Expert Systems with Applications (ESWA).
Portal: https://www.editorialmanager.com/eswa/
Article type: Research paper / full-length article (match the EM dropdown).
Review: double-anonymized. Authors appear only on the title page.

ESWA offers gold open access for a fee. Do not pick it for this submission.
Choose subscription / no gold OA so the author charge stays at USD 0.

## Clicks that keep the fee at zero

1. Publishing option: Subscription. Not gold OA. Not "publish open access."
2. Funder: No, unless a funder later requires OA.
3. Optional co-submissions (Data in Brief, MethodsX, and similar): decline unless you intend to pay and prepare those files.
4. Graphical abstract: skip unless you already have one.
5. After acceptance, Rights and Access: subscription again.
6. Print color email: web color only. Color on ScienceDirect is free. Print color is not.

Do not post this manuscript to arXiv or any preprint server before a decision while the journal's Guide for Authors still treats preprints as a conflict with double-anonymized review. Re-check the live Guide before upload.

## Files to upload

Flat folder. No subdirectories. PDF is not a source file.

| File | EM item type |
| --- | --- |
| manuscript.tex | Manuscript |
| manuscript.bbl | Manuscript, after a local TeX compile |
| references.bib | Manuscript |
| Figure_1.png, Figure_2.png | Figure |
| TitlePage.docx | Title Page |
| Highlights.docx | Highlights |
| CoverLetter.docx | Cover Letter |
| CompetingInterests.docx | Declaration of Competing Interests |

CompetingInterests.docx is a stand-in. Elsevier prefers the Word file from their declarations tool. Open the Guide for Authors, choose "I have nothing to declare," download, replace this file.

## Fill before upload

Title page: corresponding-author phone. Street address in Albany if you have one. ORCID if you want it in EM.

CRediT lists no funding role. That matches the no-grant sentence.

Macros still marked TBD in manuscript.tex need numbers from:
- benchmarks/eswa_baseline_results.json (Adaptive-k, title-prefix, MaxSim, late-encoding-style)
- benchmarks/eswa_generation_results.json (small-LLM faithfulness / accuracy)

Run `python fill_numbers.py` after those JSON files exist.

## Compile locally

```bash
cd paper/submission
tectonic manuscript.tex
```

Upload the .tex, .bib, .bbl, figures. Not the PDF as the manuscript source.

## Code and data

BEIR SciFact and NFCorpus are public. Cite Thakur et al. (2021).
Do not put github.com/AlmutazYounes in the blinded manuscript. After acceptance, point data availability at https://github.com/AlmutazYounes/qals-rag.

## Prior submission note

An earlier version was desk-rejected at Information Processing & Management for incremental contribution and missing SOTA / LLM baselines. This ESWA rewrite narrows the claim to query-time contiguous span assembly under calibrated B with a Pareto token story, adds Adaptive-k and late-chunking-style baselines, adds a generation evaluation section, and cites 2025--2026 related work. Do not submit until TBD macros are filled or explicitly accepted as pending by the submitting agent.
