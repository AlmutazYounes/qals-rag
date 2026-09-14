"""Emit TitlePage, Highlights, CoverLetter, CompetingInterests Word files for ESWA."""

from pathlib import Path
from write_docx import write_docx

ROOT = Path(__file__).resolve().parent


def main():
    write_docx(ROOT / "TitlePage.docx", [
        ("h", "Title page"),
        "Query-adaptive late segmentation: budgeted contiguous evidence assembly for retrieval-augmented generation",
        "",
        "Mohammad Bani Younes",
        "Faculty of Information Technology, Ajloun National University, P.O. Box 43, Ajloun 26810, Jordan",
        "mohamed.banyyounes@anu.edu.jo",
        "",
        "Mutaz Younes (corresponding author)",
        "Independent researcher, Albany, New York, USA",
        "mutazyounes@gmail.com",
        "Phone: [ADD CORRESPONDING-AUTHOR PHONE BEFORE UPLOAD]",
        "Postal address: [ADD ALBANY STREET ADDRESS BEFORE UPLOAD]",
        "",
        ("h", "Author contributions (CRediT)"),
        "Mohammad Bani Younes: Conceptualization, Methodology, Software, Validation, Investigation, Writing - original draft.",
        "Mutaz Younes: Supervision, Formal analysis, Writing - review and editing, Project administration.",
        "",
        ("h", "Funding"),
        "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.",
        "",
        ("h", "Declaration of competing interests"),
        "See the uploaded Declaration of Competing Interests. The authors have nothing to declare.",
        "",
        ("h", "Acknowledgements"),
        "None.",
    ])

    write_docx(ROOT / "Highlights.docx", [
        ("h", "Highlights"),
        ("b", "Query-time contiguous spans under token budget B for RAG."),
        ("b", "Pair enumeration selects at most two non-overlapping spans."),
        ("b", "Hybrid RRF leads SciFact nDCG; QALS is the low-token point."),
        ("b", "Conformal B saturates at a 2000-word cap on SciFact."),
        ("b", "Adaptive-k, late-chunk controls, and LLM eval protocol included."),
    ])

    write_docx(ROOT / "CoverLetter.docx", [
        ("h", "Cover letter"),
        "Dear Editor,",
        "",
        "Please consider our research manuscript, Query-adaptive late segmentation: budgeted contiguous evidence assembly for retrieval-augmented generation, for Expert Systems with Applications.",
        "",
        "An earlier version was desk-rejected at Information Processing and Management. The editor judged the contribution incremental relative to existing work on late chunking, contextual retrieval, and adaptive granularity, and asked for stronger SOTA and LLM-facing baselines.",
        "",
        "This rewrite answers that feedback directly. We claim only query-time contiguous span assembly under a calibrated token budget B, with a Pareto story against full-document prompts. We position against late chunking, Anthropic-style contextual retrieval, WADSeg, Adaptive-k, Mix-of-Granularity, LGMGC, LumberChunker, SmartChunk, and related 2025-2026 papers. We add Adaptive-k and late-chunking-style retrieval baselines, ablations that separate packing from multi-vector scoring, and a generation evaluation section with faithfulness and answer metrics under a shared small instruct model. Split-conformal budgeting is reported honestly, including saturation at a 2000-word cap on SciFact. Hybrid RRF remains the stronger SciFact nDCG@10 ranker in our MiniLM setup.",
        "",
        "The manuscript is written as a deployable evidence-assembly stage for expert RAG systems, which matches ESWA's applied scope. It is original, is not under review elsewhere, and has not been posted as a preprint. All authors approve this submission. We request subscription publication, not gold open access.",
        "",
        "Corresponding author: Mutaz Younes, mutazyounes@gmail.com, Albany, New York, USA. Phone: [ADD PHONE].",
        "",
        "Sincerely,",
        "Mohammad Bani Younes and Mutaz Younes",
    ])

    write_docx(ROOT / "CompetingInterests.docx", [
        ("h", "Declaration of competing interests"),
        "The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.",
        "",
        "Replace this file with the Word document from Elsevier's declarations tool before upload. Choose I have nothing to declare.",
    ])
    print("Wrote TitlePage, Highlights, CoverLetter, CompetingInterests")


if __name__ == "__main__":
    main()
