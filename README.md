# Verizon Customer Contract Default Risk Reduction

This project presents an end-to-end analytics and decision-support solution
designed to reduce customer contract default risk.
It integrates predictive modeling, business impact analysis,
and interactive product demos to support risk-aware decision making.

---

## 🔍 Problem Statement
Customer contract defaults lead to material credit losses and revenue risk.
The objective of this project is to proactively identify high-risk customers
and translate model predictions into measurable financial impact.

---

## 📊 Data & Modeling
- Binary default prediction using historical customer, payment, and contract features
- Gradient boosting classification model (XGBoost)
- Model optimization focused on recall-driven risk control

**Evaluation sample outcomes:**
- High-risk customers identified: 492
- Lower-risk customers subject to restriction: 607

---

## 💰 Business Impact Analysis
Model outputs are translated into financial outcomes using conservative assumptions.

**Sample-level net profit impact:** ~$189K

**Scenario-based annualized profit impact:**
- Conservative rollout: ~$15M
- Base case rollout: ~$45M
- Upside scenario: ~$50M+

This framing enables decision-makers to assess both upside potential
and downside risk under different deployment scopes.

![Profit Impact](images/profit_scenario.png)

---

## 📈 Executive Dashboard (Looker Studio)
An interactive dashboard built to explore:
- Default rates and customer segments
- Payment behavior and contract characteristics
- Risk concentration patterns

🔗 **Live Dashboard:** *(insert Looker Studio link here)*

![Dashboard](images/eda_dashboard.png)

---

## 🧪 Product Demo (Streamlit)
A lightweight front-end demonstrating how the model can be used in practice:
- Single customer risk assessment
- Batch scoring
- Risk-based decision output