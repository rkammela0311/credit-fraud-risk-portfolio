# Application Fraud Detection

## What this does

Flags fraudulent loan and credit-card applications at origination. The dominant pattern in 2025/2026 is **synthetic identity fraud** — fraudsters combine real elements (often a child's or deceased person's SSN) with fabricated names, addresses, phones, and emails to create a "person" who applies for credit, builds a thin file, and eventually busts out.

Synthetic identity is the fastest-growing fraud-loss category in US lending. It's particularly insidious because:
- The "borrower" passes most KYC checks (the SSN is real)
- Losses are often misclassified as credit losses (they didn't pay → charge-off), not fraud
- The fraudster can nurture the identity for years before busting out

## Key signals

The model leans heavily on **identity-element age and consistency**:

| Signal | Why it matters |
|---|---|
| `email_age_days < 30` | Synthetic identities need fresh emails — real people have years-old Gmail accounts |
| `phone_age_days < 60` | Fresh VoIP / burner numbers are a tell |
| `ssn_age_consistent` | Does the SSN issuance year line up with the stated DOB? Children's SSNs used in synthetic fraud will fail this |
| `ip_country_matches_address` | Application from a different country than the stated address |
| `months_at_address < 3` | Combined with new email/phone, suggests a freshly-built identity |
| `num_inquiries_6m > 8` | Bust-out playbook: apply everywhere at once |
| Young + high-income | Synthetic identities often overstate income to get higher limits |

A composite `identity_mismatch_score` (sum of red flags) is fed as a feature alongside the raw signals — gives the model a strong starting point and a feature that's easy to explain in adverse-action and SAR (Suspicious Activity Report) workflows.

## Why the threshold is set higher

Application fraud teams typically run at **higher recall** (90%+) than transaction fraud. The economics are different:
- Transaction false positive = customer's card declined at the cashier (high friction, lost sale)
- Application false positive = underwriter spends 10 minutes on a manual review (low friction, easily absorbed)

So we tune the threshold for 90% recall and accept lower precision.

## Output

The script reports:
- Validation and OOT metrics at the tuned threshold
- A queue-sizing table: review the top 0.5% / 1% / 2% / 5% / 10% of applications, what fraud do you capture and at what precision
- Top 10 features by gain

## Run it

```bash
python application_fraud.py
```

## What's intentionally not here

- **Bureau / consortium signals** — Equifax FraudIQ, Experian Precise ID, LexisNexis ThreatMetrix. In production these are the highest-IV features by far.
- **Device fingerprinting** — iovation, ThreatMetrix, Sift. Catches bot-driven application factories.
- **Velocity** — same-device, same-IP, same-phone applications across institutions. Done via consortium data.
- **Reject inference** — declined-fraud applicants are missing-not-at-random in any historical sample. Same problem as credit scoring.
