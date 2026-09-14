# Prior art bar (IP&M and ESWA, 2024--2026)

Jim Jansen's IP&M desk bar for RAG / LLM retrieval is not a chunking trick alone. Recent accepted papers run BEIR or TREC-style ranking, LLM rerankers or generators, ablations, and efficiency numbers. A MiniLM-only, two-corpus, retrieval-only Pareto plot sits below that bar unless QALS is framed as CPU retrieval / prompt budgeting and the baselines match that claim.

Sources checked via Crossref and publisher pages (Sep 2026). Titles, authors, years, venues, and DOIs below are from those records.

## IP&M papers (venue bar)

1. **AFR-Rank: An effective and highly efficient LLM-based listwise reranking framework via filtering noise documents.** Yinghao Xiong, Xinhui Tu, Weizhong Zhao. 2025. *Information Processing & Management* 62(6):104232. DOI: [10.1016/j.ipm.2025.104232](https://doi.org/10.1016/j.ipm.2025.104232).  
   *QALS vs this:* AFR-Rank filters then listwise-reranks first-stage hits with an LLM on TREC DL / BEIR and reports NDCG@10 plus API cost; QALS must either stay explicitly retrieval-only or add an LLM ranking / cost comparison.

2. **Towards evidence-aware retrieval-augmented generation via self-corrective chain-of-thought (SC-RAG).** Yining Li, Wenjun Ke, Jiajun Liu, Peng Wang, Jianghan Liu, Yao He. 2026. *Information Processing & Management* 63(2):104369. DOI: [10.1016/j.ipm.2025.104369](https://doi.org/10.1016/j.ipm.2025.104369).  
   *QALS vs this:* SC-RAG couples hybrid evidence retrieval with generative self-correction on LaMP and HotpotQA; QALS cannot claim a RAG contribution without an end-task or faithfulness metric beyond document nDCG.

3. **Defining the problem: The impact of OCR quality on retrieval-augmented generation performance and strategies for improvement.** Minchae Song. 2026. *Information Processing & Management* 63(1):104368. DOI: [10.1016/j.ipm.2025.104368](https://doi.org/10.1016/j.ipm.2025.104368).  
   *QALS vs this:* Song shows RAG error tracks input structure more than OCR accuracy; QALS must say it assumes clean text and that its span program is orthogonal to OCR structuring.

4. **Retriever-generator-verification: A novel approach to enhancing factual coherence in open-domain question answering.** Shiqi Sun, Kun Zhang, Jingyuan Li, Min Yu, Kun Hou, Yuanzhuo Wang, Xueqi Cheng. 2025. *Information Processing & Management* 62(4):104147. DOI: [10.1016/j.ipm.2025.104147](https://doi.org/10.1016/j.ipm.2025.104147).  
   *QALS vs this:* RGV verifies candidate answers against external knowledge; QALS's conformal \(B\) covers gold-document inclusion in the prompt, not answer factuality.

5. **Retrieve–Revise–Refine: A novel framework for retrieval of concise entailing legal article set.** Chau Nguyen, Phuong Nguyen, Le-Minh Nguyen. 2025. *Information Processing & Management* 62(1):103949. DOI: [10.1016/j.ipm.2024.103949](https://doi.org/10.1016/j.ipm.2024.103949).  
   *QALS vs this:* Both care about conciseness, but this paper uses small+large LMs to shrink legal article sets on COLIEE; QALS must stress query-time contiguous span packing under a hard token budget, not LLM revise/refine.

6. **INKER: Adaptive dynamic retrieval augmented generation with internal-external knowledge integration.** Mingjun Zhou, Jiuyang Tang, Weixin Zeng, Xiang Zhao. 2026. *Information Processing & Management* 63(3):104534. DOI: [10.1016/j.ipm.2025.104534](https://doi.org/10.1016/j.ipm.2025.104534).  
   *QALS vs this:* INKER adapts when to use internal vs external knowledge during generation; QALS adapts which contiguous sentences enter the prompt after dense sentence matching.

7. **Leveraging historical information to boost retrieval-augmented generation in conversations.** Fengran Mo, Yifan Gao, Zhuofeng Wu, Xin Liu, Pei Chen, Zheng Li, Zhengyang Wang, Xian Li, Meng Jiang, Jian-Yun Nie. 2026. *Information Processing & Management* 63(2):104449. DOI: [10.1016/j.ipm.2025.104449](https://doi.org/10.1016/j.ipm.2025.104449).  
   *QALS vs this:* Mo et al. exploit multi-turn history for conversational RAG; QALS is single-query span assembly and needs that scope stated up front.

8. **MedScaleRE-PF: a prompt-based framework with retrieval-augmented generation, chain-of-thought, and self-verification for scale-specific relation extraction in Chinese medical literature.** Zhenli Chen, Jie Hao, Haixia Sun, Liang Zhao, Jiao Li, Qing Qian, Qinglong Peng, Xuwen Wang, Shan Cong, Liu Shen, Zhen Guo, Siyue Pu, Yan Lin. 2025. *Information Processing & Management* 62(6):104278. DOI: [10.1016/j.ipm.2025.104278](https://doi.org/10.1016/j.ipm.2025.104278).  
   *QALS vs this:* This is an application RAG stack with CoT and self-verification on a clinical IE task; QALS is a general retrieval mechanism and must not sell itself as domain end-task SOTA.

## ESWA papers (RAG chunking / segmentation)

9. **WADSeg: Exploiting weak attention associations for enhanced knowledge segmentation in RAG.** Tiezheng Guo, Chen Wang, Qingwen Yang, Jiawei Tang, Yanyi Liu, Yingyou Wen. 2026. *Expert Systems with Applications* 297:129297. DOI: [10.1016/j.eswa.2025.129297](https://doi.org/10.1016/j.eswa.2025.129297).  
   *QALS vs this:* WADSeg cuts documents offline from LLM attention maps before indexing; QALS keeps sentence vectors and only builds spans after the query under budget \(B\).

10. **A comparative evaluation of the effectiveness of document splitters for large language models in legal contexts.** Mateusz Płonka, Krzysztof Kocot, Kacper Hołda, Krzysztof Daniec, Aleksander Nawrat. 2025. *Expert Systems with Applications* 272:126711. DOI: [10.1016/j.eswa.2025.126711](https://doi.org/10.1016/j.eswa.2025.126711).  
    *QALS vs this:* Płonka et al. compare fixed splitters for legal RAG corpora; QALS must beat recursive / semantic fixed cuts on ranking-vs-tokens, not only pick a better offline splitter.

11. **LLM-confidence reranker: A training-free approach for enhancing retrieval-augmented generation systems.** Zhipeng Song, Xiangyu Kong, Xinrui Bao, Yizhi Zhou, Jiulong Jiao, Sitong Liu, Yuhang Zhou, Heng Qi. 2026. *Expert Systems with Applications* 314:131627. DOI: [10.1016/j.eswa.2026.131627](https://doi.org/10.1016/j.eswa.2026.131627).  
    *QALS vs this:* LCR reranks BM25 / Contriever lists with black-box LLM confidence on BEIR and TREC and reports NDCG@5 plus QA EM; QALS's differentiation is span construction and prompt budgeting, so it needs either that comparison or a narrower claim.

## What this implies for QALS

Cite AFR-Rank, SC-RAG, and WADSeg when arguing novelty. State clearly that QALS stores title-prefixed sentence multi-vectors and assembles at most two contiguous spans at query time under \(B\), optionally calibrated by split conformal gold-inclusion depth. If resubmitting to IP&M, add at least one LLM-facing or faithfulness metric, or reframe as a CPU retrieval / prompt-budget paper with matching baselines.
