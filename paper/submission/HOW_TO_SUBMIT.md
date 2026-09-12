# Submit to IP&M without paying

Target: Elsevier Information Processing & Management.
Portal: https://www.editorialmanager.com/ipm/
Article type: Research manuscript.

Gold open access at this journal is billed at USD 3,720 plus tax. Do not pick it.

## Clicks that keep the fee at zero

1. Publishing option: Subscription. Not gold OA. Not "publish open access."
2. Funder: No.
3. Data in Brief / MethodsX co-submission: decline.
4. Graphical abstract: skip.
5. After acceptance, Rights and Access: subscription again.
6. Print color email: web color only. Color on ScienceDirect is free. Print color is not.

Do not post this manuscript to arXiv or any preprint server before a decision. IP&M is double-anonymized and the current Guide for Authors forbids a preprint until then.

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

## Compile locally

```bash
cd paper/submission
tectonic manuscript.tex
```

Upload the .tex, .bib, .bbl, figures. Not the PDF as the manuscript source.

## Code and data

BEIR SciFact and NFCorpus are public. Cite Thakur et al. (2021).
Do not put github.com/AlmutazYounes in the blinded manuscript. After acceptance, point data availability at https://github.com/AlmutazYounes/qals-rag.
