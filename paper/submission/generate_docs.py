"""Emit TitlePage, Highlights, CoverLetter, CompetingInterests Word files."""

from pathlib import Path
from write_docx import write_docx

ROOT = Path(__file__).resolve().parent


def main():
    write_docx(ROOT / "TitlePage.docx", [
        ("h", "Title page"),
        "Query-adaptive late segmentation: dynamic context assembly via sentence multi-vectors and split-conformal budgeting",
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
        "Mohammad Bani Younes: Conceptualization, Methodology, Software, Validation, Investigation, Writing – original draft.",
        "Mutaz Younes: Supervision, Formal analysis, Writing – review and editing, Project administration.",
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
        ("b", "Sentence multi-vectors postpone chunk cuts until the query is known."),
        ("b", "A 1D program selects at most two non-overlapping spans under budget B."),
        ("b", "Split-conformal B is the train quantile of gold-inclusion token depth."),
        ("b", "Hybrid RRF often leads SciFact nDCG@10 at full-document token cost."),
        ("b", "QALS stays competitive on nDCG@10 while using a much smaller prompt."),
    ])

    write_docx(ROOT / "CoverLetter.docx", [
        ("h", "Cover letter"),
        "Dear Editor,",
        "",
        "Please consider our research manuscript, Query-adaptive late segmentation: dynamic context assembly via sentence multi-vectors and split-conformal budgeting, for Information Processing & Management.",
        "",
        "Retrieval-augmented generation still cuts text into static windows and concatenates a fixed number of hits. We index title-prefixed sentence vectors, assemble at most two contiguous spans under a token budget at query time, and set that budget with split-conformal prediction on gold-inclusion depth. On BEIR SciFact and NFCorpus, with a shared MiniLM encoder, hybrid dense-lexical fusion remains a strong SciFact ranker while charging full-document prompts. QALS is a Pareto point: document nDCG@10 stays in range of character chunking, with a much smaller prompt.",
        "",
        "The work sits at the intersection of information retrieval and computing, which matches the journal's research-manuscript scope. It is original, is not under review elsewhere, and has not been posted as a preprint. All authors approve this submission. We request subscription publication, not gold open access.",
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
