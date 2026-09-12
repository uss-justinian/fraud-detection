# Fraud Detection Domain Glossary

Shared vocabulary for this platform. Every AI agent working on any component
reads this before writing a component spec — it's part of the shared
"domain context" layer described in `CONTEXT_ARCHITECTURE.md`, so that
"velocity," "chargeback," and "step-up" mean the same thing in every
component spec across the platform, not whatever an individual agent
invents locally.

| Term | Meaning |
|---|---|
| **Card-testing** | An attacker validates stolen card numbers by running many small transactions in rapid succession before attempting a large one. Shows up as a *velocity* anomaly. |
| **Velocity** | The rate of transactions/events for an entity (card, account, device, IP) over a rolling window. Core fraud signal; also core latency/cost driver, since it requires stateful lookups. |
| **Chargeback** | A forced transaction reversal initiated by the cardholder's bank, typically after the cardholder disputes a charge. The lagging, authoritative (but slow — often 30–90 days) fraud label. |
| **Friendly fraud** | A chargeback filed by the genuine cardholder despite the transaction being legitimate (e.g. "I didn't recognize the merchant name"). Pollutes chargeback data as a fraud-label source — a real system must account for this, not treat every chargeback as true fraud. |
| **Step-up authentication** | An extra verification challenge (OTP, 3-D Secure, biometric) triggered for medium-risk transactions instead of an outright decline — trades a little friction for a lot of recovered legitimate volume. |
| **False positive (in this domain)** | A legitimate transaction incorrectly declined or held for review. Has a direct, measurable revenue/customer-trust cost — unlike many ML domains, false positives here are not "free." |
| **Precision / Recall trade-off (fraud framing)** | Precision = review capacity efficiency (how many flagged transactions are worth an analyst's time); Recall = fraud caught. A platform decision, not just a modeling one — driven by analyst headcount and risk appetite. |
| **Feature store** | A system that computes and serves the same feature definitions consistently to both real-time scoring and offline training, preventing *training/serving skew* (the single most common way fraud models silently degrade in production). |
| **Champion / challenger** | Running a new ("challenger") model alongside the live ("champion") model on real traffic without it making decisions, to compare performance before promotion. |
| **Model drift** | Degradation in model performance over time, from either *data drift* (input distributions shift — e.g. a new merchant category becomes popular) or *concept drift* (the actual fraud/legitimate relationship shifts — fraud rings adapt to evade the current model). |
| **Case management** | The system and workflow fraud analysts use to review flagged transactions, investigate, and record a disposition (confirmed fraud / false positive / needs more info). |
| **Disposition** | An analyst's final judgment on a reviewed case. The primary source of *fast* labeled data (days, not the 30–90 days chargebacks take) — a key input to the feedback loop. |
| **Adverse action** | A decision (decline, account restriction) that negatively affects a customer, which in many jurisdictions carries an obligation to be explainable/contestable — a real driver for keeping rules-engine coverage and model explainability, not just chasing the highest-AUC black box. |
| **Rules engine** | A deterministic, human-authored set of if/then conditions (e.g. "decline if amount > $5,000 AND country != home_country AND account_age < 1 day"). Slower to adapt than ML, but instantly explainable and auditable — handles known fraud patterns and regulatory/compliance holds that must never depend on a probabilistic model. |
