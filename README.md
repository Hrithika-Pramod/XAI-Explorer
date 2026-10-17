# XAI Explorer — Explainable AI Demo

A small Streamlit application for demonstrating Explainable AI (XAI) concepts:
- Logistic regression trained on synthetic educational data
- Coefficient-based local feature contribution
- Counterfactual exploration
- Held-out accuracy and ROC-AUC
- Clear limitations and discussion prompts

## Important limitation
This is an educational prototype. The labels are generated from a synthetic rule and do not represent real loan outcomes. Do not use it for real lending or financial decisions. Its metrics only describe performance on a held-out split of this synthetic dataset.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install and run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens a local browser tab. Keep the terminal open while demonstrating.

## Deploy online with Streamlit Community Cloud

1. Create a GitHub repository, e.g. `xai-explorer`.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file to `app.py`.
5. Deploy. Share the generated app URL with your faculty.

Platform menus and availability can change. Never commit passwords, API keys, or private data.

## Suggested submission package
- Live app URL
- GitHub repository URL
- Report explaining XAI, logistic regression, feature contributions, counterfactuals, evaluation, limitations, and references
- 2–3 minute screen recording

## Suggested report structure
1. Abstract and problem statement
2. Explainable AI background
3. System architecture and workflow
4. Synthetic data generation and model
5. Explanation method and counterfactual method
6. Results and screenshots
7. Limitations, ethics, and future work
8. References
