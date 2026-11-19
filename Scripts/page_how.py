"""Page D: How This Project Works. A ten-point explanation for a student presenting the project.

Every figure comes from reports/metrics.json, reports/model_comparison_cv.csv or the saved model.
"""
import streamlit as st

from ui_common import esc, page_header, step

PLAIN_MODEL = {"Voting (GB+LR+Ada)": "Voting team (Gradient Boosting + Logistic Regression + AdaBoost)",
               "AdaBoost": "AdaBoost", "Logistic Regression": "Logistic Regression", "Gradient Boosting": "Gradient Boosting",
               "SVC (linear)": "Support Vector Machine (linear)", "Random Forest": "Random Forest",
               "Gaussian NB": "Naive Bayes", "Kernel SVM (rbf)": "Support Vector Machine (curved)",
               "KNN": "K-Nearest Neighbours", "Decision Tree": "Decision Tree"}

FLOW = [("Customer Data", "19 details about one customer"), ("Data Preprocessing", "clean and convert to numbers"),
        ("ML Model", "three algorithms vote"), ("Churn Probability", "a chance from 0% to 100%"),
        ("Prediction", "leave if 39% or more"), ("Retention Suggestion", "ideas to keep the customer")]

FEATURE_GROUPS = [
    ("About the customer (4)", "Gender, senior citizen, has a partner, has dependents."),
    ("Their account (6)", "Months as a customer (tenure), contract type, paperless billing, payment method, monthly bill, "
                          "total charged so far."),
    ("Services they use (9)", "Phone service, multiple lines, internet service, online security, online backup, "
                              "device protection, tech support, streaming TV, streaming movies."),
]


def flow_html():
    parts = []
    for i, (title, sub) in enumerate(FLOW):
        parts.append(f'<div class="node{" end" if i == len(FLOW) - 1 else ""}"><b>{esc(title)}</b><span>{esc(sub)}</span></div>')
        if i < len(FLOW) - 1:
            parts.append('<div class="arrow" aria-hidden="true">&rarr;</div>')
    return '<div class="cs-flow">' + "".join(parts) + "</div>"


def render(ctx):
    m, comp, calib = ctx["metrics"], ctx["comparison"], ctx["calibration"]
    op, dflt = m["test_tuned_threshold"], m["test_default_threshold"]
    cm, dcm = op["confusion_matrix"], dflt["confusion_matrix"]
    thr = op["threshold"]
    leavers = cm["tp"] + cm["fn"]
    flagged = cm["tp"] + cm["fp"]
    total = len(ctx["customers"])
    rate = float(ctx["customers"]["churned"].mean())
    removed = total - m["n_rows_after_cleaning"]

    st.markdown(page_header("How This Project Works",
                            "A step-by-step explanation, written to help explain the project in a demonstration or viva."),
                unsafe_allow_html=True)
    st.markdown("##### The big picture")
    st.markdown(flow_html(), unsafe_allow_html=True)
    st.caption("Read left to right: the customer's details go in, and a retention idea comes out.")

    st.markdown(step(1, "What does customer churn mean?",
                     "<p><b>Churn</b> means a customer stops using a company's service and leaves. In this project, a customer "
                     "who <i>left</i> is called a <b>churned</b> customer, and a customer who is still with the company "
                     "<i>stayed</i>.</p>"), unsafe_allow_html=True)
    st.markdown(step(2, "Why do telecom companies want to predict churn?",
                     "<p>Phone and internet companies have many customers who can switch provider easily. Winning a new "
                     "customer is generally more expensive than keeping an existing one. If the company can <b>spot customers "
                     "who may leave</b> early, it can offer help or a better deal <i>before</i> they go.</p>"),
                unsafe_allow_html=True)
    st.markdown(step(3, "What dataset is used?",
                     f"<p>A public telecom customer dataset with <b>{total:,} customers</b>. Each row is one "
                     f"customer, with details about their account and services, and whether they left within the last month. "
                     f"About <b>{rate:.1%}</b> of customers left. {removed} brand-new customers with no billing history were "
                     f"removed, leaving <b>{m['n_rows_after_cleaning']:,}</b> for the model.</p>"
                     f"<p>The data is split into <b>{m['n_train']:,} customers for teaching the model</b> (training) and "
                     f"<b>{m['n_test']:,} customers kept aside to test it</b>.</p>"), unsafe_allow_html=True)

    feats = "".join(f"<li><b>{esc(t)}:</b> {esc(d)}</li>" for t, d in FEATURE_GROUPS)
    st.markdown(step(4, "What customer details does the model use?",
                     f"<p>The model receives <b>19 details</b> about a customer (called <i>features</i>):</p><ul>{feats}</ul>"
                     "<p>These are exactly the fields on the Predict page.</p>"), unsafe_allow_html=True)
    st.markdown(step(5, "How is the data prepared (preprocessing)?",
                     "<p>Computers need numbers, so the data is prepared before the model sees it:</p><ul>"
                     "<li><b>Clean:</b> remove the 11 customers with no billing history yet.</li>"
                     "<li><b>Turn words into numbers:</b> choices such as <i>Contract = One year</i> become 0/1 columns "
                     "(one-hot encoding).</li>"
                     "<li><b>Put numbers on the same scale:</b> months, monthly bill and total charged are standardised, "
                     "so a large number does not dominate (scaling).</li>"
                     "<li><b>Set aside test customers first:</b> the test customers are separated <i>before</i> any "
                     "learning, so the model is judged on customers it has never seen. This avoids &lsquo;cheating&rsquo; "
                     "(data leakage).</li></ul>"), unsafe_allow_html=True)

    rows = "".join(f"<tr><td>{esc(PLAIN_MODEL.get(r['Model'], r['Model']))}</td><td>{r['CV roc_auc']:.3f}</td></tr>"
                   for _, r in comp.sort_values("CV roc_auc", ascending=False).iterrows()) if comp is not None else ""
    table = (f"<table style='width:100%;border-collapse:collapse'><thead><tr><th align='left'>Algorithm</th>"
             f"<th align='left'>Score (higher is better, 1.0 = perfect)</th></tr></thead><tbody>{rows}</tbody></table>"
             if rows else "")
    st.markdown(step(6, "Why were several machine-learning algorithms compared?",
                     "<p>No single algorithm is best for every problem, so ten were tried on the same data and tested the same "
                     "way. The score below (ROC-AUC) measures how well each one ranks customers who left above those who "
                     "stayed. It was measured on the training customers only, using cross-validation.</p>" + table),
                unsafe_allow_html=True)
    st.markdown(step(7, "How does the final model predict churn?",
                     "<p>The final model is a <b>voting team</b> of three algorithms: Gradient Boosting, Logistic Regression and "
                     "AdaBoost. Each one gives its own chance that the customer will leave, and the team <b>averages "
                     "them</b> into one number between 0% and 100%. This is the <b>churn probability</b>. The team was chosen "
                     "because it had the best cross-validated score, although Gradient Boosting alone was almost "
                     "identical.</p>"), unsafe_allow_html=True)

    st.markdown(step(8, f"What does the {thr:.0%} threshold mean?",
                     f"<p>The probability is only a number; we still need a yes/no decision. The <b>threshold</b> is the alert "
                     f"line: if the chance of leaving is <b>{thr:.0%} or more</b>, the customer is labelled <i>likely to "
                     f"leave</i>.</p><p>Why not 50%? Missing a customer who leaves is usually worse than checking on one who "
                     f"would have stayed. A lower line catches more leavers but raises more false alarms. It was chosen using "
                     f"the training data only, not the test customers.</p>"
                     f"<table style='width:100%;border-collapse:collapse'><thead><tr><th align='left'>Alert line</th>"
                     f"<th align='left'>Leavers found</th><th align='left'>False alarms</th></tr></thead><tbody>"
                     f"<tr><td><b>{thr:.0%} (used)</b></td><td>{op['recall']:.0%} ({cm['tp']} of {leavers})</td>"
                     f"<td>{cm['fp']}</td></tr>"
                     f"<tr><td>50%</td><td>{dflt['recall']:.0%} ({dcm['tp']} of {dcm['tp'] + dcm['fn']})</td>"
                     f"<td>{dcm['fp']}</td></tr></tbody></table>"), unsafe_allow_html=True)

    st.markdown(step(9, "How well does the model work? (evaluation in simple words)",
                     f"<p>These results come from the <b>{m['n_test']:,} test customers</b> the model never saw, using the "
                     f"{thr:.0%} alert line:</p><ul>"
                     f"<li><b>Accuracy {op['accuracy']:.1%}:</b> of all customers, how many were classified correctly. "
                     f"Careful: always saying &ldquo;will stay&rdquo; would already score {m['majority_baseline_accuracy_test']:.1%}, "
                     f"so accuracy alone is not enough.</li>"
                     f"<li><b>Recall {op['recall']:.1%}:</b> of customers who actually left, how many did the model identify? "
                     f"({cm['tp']} of {leavers}.)</li>"
                     f"<li><b>Precision {op['precision']:.1%}:</b> of customers predicted to leave, how many actually left? "
                     f"({cm['tp']} of {flagged}.)</li>"
                     f"<li><b>F1 score {op['f1']:.2f}:</b> one number that balances precision and recall.</li>"
                     f"<li><b>ROC-AUC {op['roc_auc']:.3f}:</b> how well the model ranks leavers above stayers "
                     f"(0.5 is guessing, 1.0 is perfect).</li></ul>"
                     f"<div class='cs-grid2'><div class='ok'><b>{cm['tn']:,}</b><span>correctly said &ldquo;will stay&rdquo;</span></div>"
                     f"<div class='mid'><b>{cm['fp']:,}</b><span>false alarms (said &ldquo;leave&rdquo;, but stayed)</span></div>"
                     f"<div class='bad'><b>{cm['fn']:,}</b><span>missed leavers (said &ldquo;stay&rdquo;, but left)</span></div>"
                     f"<div class='ok'><b>{cm['tp']:,}</b><span>correctly caught leavers</span></div></div>"),
                unsafe_allow_html=True)

    st.markdown(step(10, "What are the limitations?",
                     "<ul>"
                     "<li><b>One public dataset.</b> The model learned from these customers and may not match another company's.</li>"
                     "<li><b>Predictions are estimates, not guarantees.</b> Customers below the alert line can still leave.</li>"
                     f"<li><b>Probabilities run a little high.</b> On the test customers the model's average predicted chance of "
                     f"leaving was {calib['mean_predicted']:.1%}, while {calib['observed']:.1%} really left.</li>"
                     "<li><b>Patterns are not causes.</b> The charts and &ldquo;what stands out&rdquo; notes describe past "
                     "customers; they do not prove why someone leaves.</li>"
                     "<li><b>Retention ideas are simple rules,</b> not model output, and have not been tested on real customers.</li>"
                     "<li><b>One snapshot in time.</b> Customer behaviour changes, so a real system would need retraining.</li>"
                     "<li><b>Not for real decisions.</b> This is an academic prototype.</li></ul>"), unsafe_allow_html=True)
