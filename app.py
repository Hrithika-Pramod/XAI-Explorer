import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="XAI Explorer", page_icon="🧠", layout="wide")

# -----------------------------
# Train a small illustrative model on synthetic data.
# This is an educational demonstration, NOT a real lending model.
# -----------------------------
@st.cache_resource
def train_demo_model():
    rng = np.random.default_rng(42)
    n = 5000
    income = rng.integers(15, 151, n)            # thousands of currency units/year
    credit = rng.integers(300, 851, n)            # illustrative credit score
    debt = rng.integers(0, 101, n)                # debt burden, 0-100
    employment = rng.integers(0, 2, n)            # 0: less stable, 1: stable

    # Synthetic rule used only to create example labels.
    logit = (
        -5.0
        + 0.035 * income
        + 0.010 * (credit - 300)
        - 0.045 * debt
        + 0.9 * employment
    )
    probability = 1 / (1 + np.exp(-logit))
    labels = (rng.random(n) < probability).astype(int)

    X = pd.DataFrame({
        "Annual income (₹ lakh)": income / 10,
        "Credit score": credit,
        "Debt burden (%)": debt,
        "Stable employment": employment
    })
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    return model, X_train, metrics

model, X_train, metrics = train_demo_model()
feature_names = list(X_train.columns)

st.title("🧠 XAI Explorer")
st.caption("Why did AI decide that? — an interactive Explainable AI learning demo")

st.warning(
    "Educational prototype only. The model is trained on synthetic data and must not be "
    "used for real loan, credit, or financial decisions."
)

with st.sidebar:
    st.header("Applicant profile")
    income = st.slider("Annual income (₹ lakh)", 1.5, 15.0, 7.0, 0.5)
    credit = st.slider("Credit score", 300, 850, 680, 10)
    debt = st.slider("Debt burden (%)", 0, 100, 30, 5)
    employment_label = st.selectbox("Employment stability", ["Less stable", "Stable"])
    employment = 1 if employment_label == "Stable" else 0
    st.divider()
    st.caption("Change any input to see how the model's prediction and explanation change.")

input_df = pd.DataFrame([{
    "Annual income (₹ lakh)": income,
    "Credit score": credit,
    "Debt burden (%)": debt,
    "Stable employment": employment
}])

prob = float(model.predict_proba(input_df)[0, 1])
pred = int(prob >= 0.5)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model output", "Eligible-like" if pred else "Review-like")
with col2:
    st.metric("Predicted probability", f"{prob:.1%}")
with col3:
    st.metric("Demo ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.subheader("1. What influenced this prediction?")
st.write(
    "This explanation uses the trained logistic-regression model's coefficients. "
    "Each bar shows the signed contribution of an input relative to the training-set mean. "
    "Positive values push the model toward class 1; negative values push it toward class 0."
)

# Approximate additive log-odds contribution relative to training mean.
means = X_train.mean()
coefs = model.coef_[0]
contributions = (input_df.iloc[0] - means) * coefs
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": [float(contributions[name]) for name in feature_names]
}).sort_values("Contribution")

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ["#D95F59" if v < 0 else "#168A72" for v in contrib_df["Contribution"]]
ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Contribution to model log-odds (relative to average input)")
ax.set_title("Local feature contribution")
ax.grid(axis="x", alpha=0.2)
st.pyplot(fig, clear_figure=True)

with st.expander("Read the explanation"):
    for _, row in contrib_df.sort_values("Contribution", ascending=False).iterrows():
        direction = "pushes toward the positive class" if row["Contribution"] > 0 else "pushes away from the positive class"
        st.write(f"**{row['Feature']}**: {row['Contribution']:+.3f} — {direction}.")

st.subheader("2. Counterfactual explorer")
st.write(
    "A counterfactual asks: *What small input change could flip the model's output?* "
    "This is a mathematical demonstration, not advice about changing real financial behaviour."
)
counterfactuals = []
candidate_ranges = {
    "Annual income (₹ lakh)": np.arange(1.5, 15.01, 0.5),
    "Credit score": np.arange(300, 851, 10),
    "Debt burden (%)": np.arange(0, 101, 5),
    "Stable employment": np.array([0, 1])
}
current_values = input_df.iloc[0].to_dict()
for feature in feature_names:
    for value in candidate_ranges[feature]:
        candidate = current_values.copy()
        candidate[feature] = float(value)
        if feature == "Stable employment":
            candidate[feature] = int(value)
        candidate_df = pd.DataFrame([candidate], columns=feature_names)
        candidate_pred = int(model.predict_proba(candidate_df)[0, 1] >= 0.5)
        if candidate_pred != pred:
            delta = abs(float(value) - float(current_values[feature]))
            counterfactuals.append((feature, value, delta, candidate_pred))
            break

if counterfactuals:
    cf_df = pd.DataFrame(counterfactuals, columns=["Changed feature", "First tested value that flips output", "Absolute change", "New class"])
    cf_df["New class"] = cf_df["New class"].map({0: "Review-like", 1: "Eligible-like"})
    st.dataframe(cf_df, use_container_width=True, hide_index=True)
else:
    st.info("No single-feature flip was found in the tested ranges for this profile.")

st.subheader("3. Model transparency")
m1, m2 = st.columns(2)
with m1:
    st.write("**Model used**")
    st.write("Logistic Regression")
    st.write("**Training data**")
    st.write("5,000 synthetically generated examples")
    st.write("**Explanation method**")
    st.write("Coefficient-based local contribution")
with m2:
    st.write("**Held-out accuracy**")
    st.write(f"{metrics['accuracy']:.3f}")
    st.write("**Held-out ROC-AUC**")
    st.write(f"{metrics['roc_auc']:.3f}")
    st.write("**Important limitation**")
    st.write("Synthetic-data performance does not establish real-world validity or fairness.")

with st.expander("Key concepts "):
    st.markdown("""
    - **Explainable AI (XAI):** methods that help people understand model behaviour.
    - **Black-box model:** a model whose internal reasoning is difficult to interpret directly.
    - **Feature contribution:** how an input shifts a model's output relative to a reference.
    - **Counterfactual explanation:** a nearby input change that changes a prediction.
    - **Transparency is not fairness:** an understandable model can still be biased or inaccurate.
    - **Explanation is not causation:** a model contribution does not prove that changing a real-world factor will cause the same outcome.
    """)

st.caption("XAI Explorer • Student demonstration project")
