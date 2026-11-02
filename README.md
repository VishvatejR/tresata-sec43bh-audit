# Neuro-Symbolic Audit Engine for Indian Section 43B(h) Compliance

**Tresata Take-Home Assignment — Working Prototype & 12-Case Benchmark**

This repository contains a minimal, self-contained prototype demonstrating a **Hybrid Neuro-Symbolic Architecture** for auditing corporate trade payables under **Section 43B(h) of the Indian Income Tax Act, 1961** read with **Sections 2(b) and 15 of the MSMED Act, 2006** (Form 3CD Clause 22).

## 1. Architectural Philosophy: Why Hybrid AI + Deterministic Code?

During statutory tax audits, Chartered Accountants (CAs) must inspect unstructured vendor bundles (Udyam Certificates, Purchase Orders, Tax Invoices, Delivery Challans/GRNs, and Quality-Dispute Emails) to determine whether unpaid year-end balances to Micro and Small Enterprises (MSEs) must be disallowed as tax deductions.

* **Why Pure Rules / SQL Fail:** Critical legal facts live in unstructured text—such as whether a buyer raised a *written quality objection within 15 days of delivery* in an email thread (which legally pauses and resets the statutory payment clock), whether credit terms were agreed *bilaterally* in a signed PO versus *unilaterally* printed on an invoice footer, or whether an invoice is *OpEx* (covered) vs. *CapEx* (exempt).

* **Why a Single LLM Call Fails:** LLMs hallucinate calendar arithmetic, confuse the March 31 fiscal year-end cutoff with the October 31 Income Tax Return filing date, and cannot provide an auditable guarantee of exact repeatability.

### Our Separation of Concerns

1. **Stage 1 — Probabilistic Semantic Sensor (LLM):** Reads the PII-masked document bundle and extracts semantic facts into a strict `EvidencePacket` JSON schema (`enterprise_tier`, `primary_activity`, `nic_code_2digit`, `expense_nature`, `has_bilateral_written_agreement`, `has_written_quality_objection`, dates, verbatim `evidence_quote`, and `extraction_confidence`). **The LLM performs zero date math and never sees confidential rupee invoice amounts.**

2. **Stage 2 — Deterministic Statutory & Calendar Engine (Pure Python):** Verifies that the LLM's verbatim quote exists in the source text, applies version-controlled statutory rules (`IN_IT_ACT_43BH_AY2026_27_v1.2`), computes exact appointed-day deadlines (`15 days` default vs. `min(PO_days, 45 days)`), evaluates defect-resolution clock resets, joins the unmasked rupee amount from the ERP ledger, and routes low-confidence or ambiguous cases to a **Human Chartered Accountant**.

## 2. How to Run the Prototype

### Prerequisites

* **Python 3.9+** (Uses only the Python Standard Library—zero external `pip` packages required).

* **Optional (For Live LLM Extraction):** A free-tier API key from [OpenRouter](https://openrouter.ai/). By default, the script runs the full **12-case benchmark offline** using pre-captured LLM extraction fixtures, and switches to live OpenRouter API calls (`meta-llama/llama-3.3-70b-instruct:free`) when `--live` is passed with an `OPENROUTER_API_KEY`.

### Step 1: Run the 12-Example Benchmark (Zero-Config Offline Mode)

```
python3 sec43bh_audit_prototype.py


```

### Step 2: Run with Live OpenRouter Free-Tier LLM Calls

```
export OPENROUTER_API_KEY="sk-or-v1-your-free-key-here"
python3 sec43bh_audit_prototype.py --live --model "meta-llama/llama-3.3-70b-instruct:free"


```

*(Note: Free-tier OpenRouter models enforce rate limits, so the script includes a polite 2-second delay between test cases when running in `--live` mode.)*

## 3. Summary of the 12 Synthetic Test Examples

All 12 test cases are synthetic representations modeled on real Indian manufacturing procurement bundles across the **March 31, 2026 (FY 2025–26 / AY 2026–27)** fiscal cutoff:

| Case ID | Scenario Description | Vendor Tier & NIC | Delivery / Invoice Date | Terms & Dispute Facts | Actual Paid Date | Expected Ground Truth | Prototype Output | Result | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **TC-01** | **PO "Net 60" Statutory Cap:** Signed PO grants 60 days, but Sec 15 caps written agreements at 45 days. | Small, Mfg (NIC 25) | 2026-02-01 | Signed PO: 60d -> capped at 45d (Due 2026-03-18). | 2026-04-02 | `DISALLOWED_43BH` | `DISALLOWED_43BH` | **PASS** | 
| **TC-02** | **Unilateral Invoice Footer:** Invoice says "Net 30 Days" but no signed PO exists. Defaults to 15-day rule. | Micro, Mfg (NIC 28) | 2026-03-10 | Unilateral 30d ignored -> 15d cap (Due 2026-03-25). | 2026-04-05 | `DISALLOWED_43BH` | `DISALLOWED_43BH` | **PASS** | 
| **TC-03** | **Year-End Straddle Paid on Time:** Delivered March 20 under 45d MSA; paid in April within statutory window. | Small, Svc (NIC 62) | 2026-03-20 | Signed MSA: 45d (Due 2026-05-04). | 2026-04-25 | `ALLOWABLE` | `ALLOWABLE` | **PASS** | 
| **TC-04** | **Valid Quality Objection (<=15d):** Delivered Feb 10; written defect email sent Feb 18 (Day 8); resolved Mar 22. | Micro, Mfg (NIC 27) | Deliv: 2026-02-10, Resolved: 2026-03-22 | Clock resets to Mar 22 + 30d PO = Due 2026-04-21. | 2026-04-18 | `ALLOWABLE` | `ALLOWABLE` | **PASS** | 
| **TC-05** | **Belated Quality Objection (>15d):** Delivered Feb 1; defect email sent Feb 22 (Day 21). Clock does not reset. | Small, Mfg (NIC 24) | 2026-02-01 | Objection >15d ignored. Due date stays 2026-03-03. | 2026-04-10 | `DISALLOWED_43BH` | `DISALLOWED_43BH` | **PASS** | 
| **TC-06** | **Udyam Wholesale Trader Exemption:** Vendor has Udyam certificate under NIC 46 (Wholesale Trade). | Small, Trading (NIC 46) | 2026-01-15 | Exempt per MSME OM dated 02.07.2021. | Unpaid | `ALLOWABLE` | `ALLOWABLE` | **PASS** | 
| **TC-07** | **Medium Enterprise Exemption:** Udyam certificate classifies vendor as a Medium Enterprise. | Medium, Mfg (NIC 29) | 2026-01-10 | Sec 43B(h) applies only to Micro and Small tiers. | Unpaid | `ALLOWABLE` | `ALLOWABLE` | **PASS** | 
| **TC-08** | **Capital Expenditure (CapEx) Exclusion:** Purchase of CNC Lathe Machine (capitalized asset, not P&L expense). | Small, Mfg (NIC 28) | 2026-01-20 | CapEx not claimed under Sec 30–37 P&L deduction. | 2026-05-10 | `ALLOWABLE` | `ALLOWABLE` | **PASS** | 
| **TC-09** | **Late Payment Settled Within Same FY:** Delivered Jan 1 (Due Jan 16), paid 71 days late on March 28. | Micro, Svc (NIC 71) | 2026-01-01 | Paid on or before March 31 fiscal cutoff -> Deductible. | 2026-03-28 | `ALLOWABLE` | `ALLOWABLE` | **PASS** | 
| **TC-10** | **Composite Supply Contract:** Single invoice mixes traded solar panels (70%) and installation services (30%). | Small, Composite | 2026-02-05 | LLM flags `COMPOSITE` activity -> routes to Human CA. | Unpaid | `ROUTE_TO_HUMAN_CA` | `ROUTE_TO_HUMAN_CA` | **PASS** | 
| **TC-11** | **Adversarial Failure 1 — Numeric Date Ambiguity:** Challan reads `03/02/2026` (intended Feb 3 in DD/MM/YYYY). | Micro, Mfg (NIC 25) | True: 2026-02-03, Parsed: 2026-03-02 | LLM parsed US `MM/DD/YYYY` (Mar 2) -> Due date shifted to Apr 1. | 2026-04-01 | `DISALLOWED_43BH` | `ALLOWABLE` *(False Negative)* | **FAIL** | 
| **TC-12** | **Adversarial Failure 2 — Dual-NIC Udyam Certificate:** Udyam lists NIC 25 (Mfg) & NIC 46 (Trading); invoice says "Industrial Valves". | Small, Dual NIC (25 & 46) | 2026-01-25 | Vendor traded imported valves, but LLM picked top NIC 25 (Mfg). | Unpaid | `ALLOWABLE` | `DISALLOWED_43BH` *(False Positive)* | **FAIL** | 

## 4. Honest Failure Analysis (Where the Prototype Breaks)

Our evaluation achieved **10/12 exact matches (83.3% accuracy)** and surfaced **2 instructive failures** (`TC-11` and `TC-12`) in Stage 1 extraction:

1. **Failure 1 (`TC-11` — False Negative from `DD/MM/YYYY` vs. `MM/DD/YYYY` Date Ambiguity):**

   * **Root Cause:** The delivery challan contained an all-numeric date `03/02/2026` (February 3, 2026 in Indian format), while the supplier's tax invoice was issued late on `02-Mar-2026`. Because `March 2` was semantically close to `02-Mar-2026`, the LLM normalized `03/02/2026` as `2026-03-02` (US `MM/DD/YYYY` format). This pushed the 30-day statutory deadline from March 5 to April 1, causing the April 1 payment to be marked `ALLOWABLE` instead of `DISALLOWED_43BH`.

   * **Remediation for v2:** Require the LLM to output both the `raw_date_string` (`"03/02/2026"`) and a boolean `is_numeric_ambiguous_date` whenever both day and month components are `<= 12`, and cross-check the extracted delivery date against the internal ERP GRN posting timestamp.

2. **Failure 2 (`TC-12` — False Positive on Dual-Activity Udyam Certificates):**

   * **Root Cause:** Many Indian engineering vendors hold a single Udyam certificate listing *both* Manufacturing (NIC 25) and Wholesale Trade (NIC 46). In `TC-12`, the vendor supplied imported third-party valves (a trading activity exempt from Section 43B(h)), but the invoice description merely read *"Supply of High-Pressure Industrial Valves"*. The LLM selected the first listed activity on the Udyam PDF (`MANUFACTURING`, NIC 25) with `0.91` confidence, resulting in a false-positive disallowance.

   * **Remediation for v2:** Change `nic_code_2digit` in `EvidencePacket` from a single string to a list `all_udyam_nic_codes: List[str]`. In Stage 2, if a vendor's Udyam certificate contains *both* manufacturing/service codes and trading codes (`45`, `46`, `47`), deterministically route the invoice to `ROUTE_TO_HUMAN_CA` unless the invoice contains an explicit manufacturing excise/HSN declaration.

## 5. Repository Structure

* `sec43bh_audit_prototype.py` — Self-contained Neuro-Symbolic Audit Engine, OpenRouter API client, Grounding Verifier, and 12-Case Benchmark Runner.

* `README.md` — Architecture documentation, execution instructions, benchmark results, and failure analysis.
