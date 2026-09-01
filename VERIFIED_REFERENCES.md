# Verified References

Every entry below was checked against a live search and its source URL/DOI
resolves to something real. This is the reference list actually used in
`paper_final.md` / `paper_final.tex`.

1. A. Shevrin, O. Peleg, et al., "Detecting Multi-Step IAM Attacks in AWS
   Environments via Model Checking," *32nd USENIX Security Symposium*, 2023.
   https://www.usenix.org/conference/usenixsecurity23/presentation/shevrin

2. T. van Ede, N. Khasuntsev, B. Steen, A. Continella, "Detecting Anomalous
   Misconfigurations in AWS Identity and Access Management Policies," *ACM
   Cloud Computing Security Workshop (CCSW)*, 2022.
   https://dl.acm.org/doi/10.1145/3560810.3564264

3. J. Wen, Z. Chen, F. Sarro, Z. Zhu, Y. Liu, H. Ping, S. Wang, "LLM-Based
   Misconfiguration Detection for AWS Serverless Computing," arXiv, 2024.
   https://arxiv.org/pdf/2411.00642

4. "Adaptive Security Policy Management in Cloud Environments Using
   Reinforcement Learning," arXiv, 2025. https://arxiv.org/pdf/2505.08837

5. "Optimizing Cybersecurity Incident Response via Adaptive Reinforcement
   Learning," published research paper, 2025.
   https://www.researchgate.net/publication/390130591

6. OWASP Foundation, "OWASP API Security Top 10," 2023.
   https://owasp.org/API-Security/

7. NIST Special Publication 800-207, "Zero Trust Architecture," National
   Institute of Standards and Technology, 2020.
   https://csrc.nist.gov/pubs/sp/800/207/final

8. Thales, "2024 Cloud Security Study," reported via Infosecurity Magazine.
   https://www.infosecurity-magazine.com/news/cloud-breaches-half-organizations/

9. IEEE DataPort, "Misconfiguration Detection Dataset for Cloud IAM and
   APIs."
   https://ieee-dataport.org/documents/misconfiguration-detection-dataset-cloud-iam-and-apis

10. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1,
    pp. 5-32, 2001. Foundational method paper for the classifier used in
    Experiments 1 and 2. DOI: 10.1023/A:1010933404324

11. F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," *IEEE ICDM*,
    2008, pp. 413-422. Foundational method paper for the classifier used
    in Experiment 3. DOI: 10.1109/ICDM.2008.17

12. A. Rahman, S. I. Shamim, D. B. Bose, R. Pandita, "Security
    Misconfigurations in Open Source Kubernetes Manifests: An Empirical
    Study," *ACM Transactions on Software Engineering and Methodology*,
    vol. 32, no. 4, 2023. Verified as a real, cited paper (appears in
    another paper's bibliography, arXiv:2512.11940).

13. Cloud Security Alliance, "Top Threats to Cloud Computing 2024," 2024.
    https://cloudsecurityalliance.org/artifacts/top-threats-to-cloud-computing-2024
    Confirmed real via multiple independent secondary sources (Help Net
    Security, BusinessWire, ChannelFutures) reporting the same report.

14. Salesforce, "Cloudsplaining," open-source software, Apache-2.0.
    https://github.com/salesforce/cloudsplaining -- this is also the
    actual DATA SOURCE for the Experiment 1 real-world validation (see
    below), not just a citation.

15. A. Alvarez, "TrailDiscover," open-source dataset.
    https://github.com/adanalvarez/TrailDiscover -- actual data source
    for the Experiment 3 grounding check (see below).

## Checked and confirmed FABRICATED -- do not cite

- "S. Challa et al., 'CloudTrail Log Anomaly Detection Using Recurrent
  Neural Networks,' IEEE Access, vol. 8, pp. 128450-128461, 2020." No
  paper with this title, these authors, or this IEEE Access volume/page
  range could be found anywhere. Removed from the reference list.

## Checked and confirmed real but not currently cited

- Palo Alto Networks Unit 42 Cloud Threat Report series is real, but no
  report matches the exact composite title "Cloud Threat Report: Incident
  Response Trends, 2023" that appeared in the earliest draft. Not cited --
  the Thales + CSA citations (8, 13) already cover the underlying claim
  with sources that check out exactly as stated.
- M. Koroma, A. Mansaray, et al., "Enhancing Cybersecurity in IoT & Cloud:
  Anomaly Detection via Ensemble Machine Learning," *Journal of Software
  Engineering and Applications*, vol. 18, no. 6, 2025. Confirmed real
  (https://file.scirp.org/pdf/jsea_9303408.pdf) but not currently cited --
  available if you want additional CloudTrail/IoT related-work coverage.

## Real-world data sources used directly in the experiments (not just cited)

This project uses two open-source projects as actual data sources for
external validation, not just as literature citations:

- **Cloudsplaining** (`salesforce/cloudsplaining`, Apache-2.0) -- its test
  fixture `test/files/example-authz-details.json` contains the real,
  currently-published JSON documents of 30 genuine AWS-managed IAM
  policies. `src/exp1_realworld_validation.py` loads these directly and
  they are the actual "real-world" test set behind Table 2 in the paper.
  The policy documents themselves are AWS's own public IAM definitions,
  not Cloudsplaining's intellectual property; Cloudsplaining is credited
  as the source of the compiled fixture.
- **TrailDiscover** (`adanalvarez/TrailDiscover`) -- its `docs/events.csv`
  contains 381 real CloudTrail event names with an incident-report-linked
  `usedInWild` flag. Used to cross-check the realism of Experiment 3's
  synthetic "rare event" category (see paper Section 3.3 / Limitations).


