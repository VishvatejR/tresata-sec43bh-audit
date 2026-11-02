#!/usr/bin/env python3
"""
Tresata Take-Home Assignment Prototype
Hybrid Neuro-Symbolic Audit Engine for Indian Income Tax Section 43B(h) & MSMED Act Section 15

Architecture Overview:
  - Stage 0 (Privacy Pre-Processor): Regex-based PII & financial price masking before LLM API calls.
  - Stage 1 (Probabilistic Semantic Sensor): Extracts unstructured temporal, dispute, and activity facts
    from Udyam PDFs, POs, Invoices, GRNs, and Email threads into a strict JSON EvidencePacket.
    Enforces substring grounding verification against the raw document text.
  - Stage 2 (Deterministic Statutory Engine): 100% pure Python rules & calendar arithmetic engine
    governed by a versioned Assessment Year configuration (STATUTORY_RULES_AY_2026_27).
    Computes exact Appointed Day resets, 15/45-day caps, March 31 fiscal cutoffs, and Rupee disallowances.

Usage:
  1. Offline Benchmark (Default - zero API key required, evaluates all 12 test bundles):
     python3 sec43bh_audit_prototype.py

  2. Live OpenRouter Free-Tier Extraction Mode:
     export OPENROUTER_API_KEY="sk-or-v1-..."
     python3 sec43bh_audit_prototype.py --mode live --model meta-llama/llama-3.3-70b-instruct:free

  3. Export Full Audit Working Paper JSON:
     python3 sec43bh_audit_prototype.py --export-json audit_working_papers_AY26_27.json
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


# ============================================================================
# 1. VERSIONED STATUTORY KNOWLEDGE BASE (Lives in Config, Never in LLM Weights)
# ============================================================================
STATUTORY_RULES_AY_2026_27: Dict[str, Any] = {
    "rule_version": "IN_IT_ACT_43BH_AY2026_27_v1.2",
    "effective_fy_start": "2025-04-01",
    "effective_fy_end": "2026-03-31",
    "default_no_agreement_days": 15,
    "max_written_agreement_days": 45,
    "objection_window_days": 15,
    "eligible_enterprise_tiers": ["MICRO", "SMALL"],
    "eligible_activities": ["MANUFACTURING", "SERVICES"],
    "excluded_nic_prefixes": ["45", "46", "47"],  # Wholesale/Retail Trade (MSME OM 02.07.2021)
    "corporate_tax_rate_effective": 0.25168,     # Sec 115BAA: 22% base + 10% surcharge + 4% cess
    "confidence_threshold": 0.85,
}


# ============================================================================
# 2. DATA CONTRACTS (Strict Boundary Between Stage 1 AI and Stage 2 Math)
# ============================================================================
@dataclass
class EvidencePacket:
    """Output of Stage 1 (LLM Semantic Extractor). Contains ZERO math or tax rulings."""
    case_id: str
    enterprise_tier: str                  # MICRO | SMALL | MEDIUM | UNKNOWN
    primary_activity: str                 # MANUFACTURING | SERVICES | TRADING | COMPOSITE
    nic_code_2digit: str                  # e.g., "25", "46", "62"
    expense_nature: str                   # OPEX | CAPEX | AMBIGUOUS
    invoice_date: str                     # YYYY-MM-DD
    delivery_date: str                    # YYYY-MM-DD
    has_bilateral_written_agreement: bool
    agreed_credit_days: Optional[int]     # None if no bilateral agreement
    has_written_quality_objection: bool
    objection_date: Optional[str]         # YYYY-MM-DD if objection raised
    defect_resolution_date: Optional[str] # YYYY-MM-DD if objection resolved
    evidence_quote: str                   # Verbatim substring from source bundle
    extraction_confidence: float          # 0.0 to 1.0


@dataclass
class AuditDecision:
    """Output of Stage 2 (Deterministic Statutory Engine)."""
    case_id: str
    scenario_name: str
    vendor_name: str
    invoice_amount_inr: float
    status: str                           # ALLOWABLE | DISALLOWED_43BH | ROUTE_TO_HUMAN_CA
    expected_ground_truth: str
    matched_ground_truth: bool
    disallowed_amount_inr: float
    cash_tax_impact_inr: float
    appointed_day_start: Optional[str]
    statutory_due_date: Optional[str]
    actual_payment_date: Optional[str]
    days_allowed: Optional[int]
    rule_trace: List[str]
    human_review_reason: Optional[str]
    rule_version: str


@dataclass
class SyntheticTestBundle:
    """Self-contained multi-document audit test case."""
    case_id: str
    scenario_name: str
    vendor_name: str
    invoice_amount_inr: float
    actual_payment_date: Optional[str]
    is_opening_balance_pre_fy: bool
    has_unexplained_credit_note: bool
    expected_ground_truth: str
    raw_document_bundle: str
    llm_extracted_fixture: EvidencePacket
    failure_analysis_note: Optional[str] = None


# ============================================================================
# 3. STAGE 0 & 1: PII SCRUBBER + PROBABILISTIC LLM EXTRACTOR (OpenRouter API)
# ============================================================================
SYSTEM_PROMPT = """You are a statutory tax audit document parser for Indian Section 43B(h) compliance.
Read the unstructured procurement document bundle and extract ONLY factual semantic attributes into
strict JSON matching the EvidencePacket schema:
{
  "case_id": "string",
  "enterprise_tier": "MICRO | SMALL | MEDIUM | UNKNOWN",
  "primary_activity": "MANUFACTURING | SERVICES | TRADING | COMPOSITE",
  "nic_code_2digit": "2-digit string e.g. 25, 46, 62",
  "expense_nature": "OPEX | CAPEX | AMBIGUOUS",
  "invoice_date": "YYYY-MM-DD",
  "delivery_date": "YYYY-MM-DD",
  "has_bilateral_written_agreement": boolean,
  "agreed_credit_days": integer or null,
  "has_written_quality_objection": boolean,
  "objection_date": "YYYY-MM-DD or null",
  "defect_resolution_date": "YYYY-MM-DD or null",
  "evidence_quote": "Exact verbatim substring copied from the input bundle (at least 15 chars)",
  "extraction_confidence": float between 0.0 and 1.0
}
CRITICAL RULES:
1. DO NOT calculate due dates, day counts, or tax disallowances.
2. has_bilateral_written_agreement is true ONLY if a signed PO or Master Service Agreement specifies credit terms. Unilateral invoice footer text ("Net 30 days on invoice") without a PO is false.
3. has_written_quality_objection is true ONLY if the buyer complained in writing about physical defects, damaged goods, or incomplete service delivery. Requests for payment extensions are false.
4. evidence_quote MUST be an exact verbatim substring from the document text."""


def scrub_sensitive_financial_pii(raw_text: str) -> str:
    """Masks Indian PAN, GSTIN, Bank Account Numbers, and INR amounts before external LLM call."""
    scrubbed = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", "[PAN_MASKED]", raw_text)
    scrubbed = re.sub(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", "[GSTIN_MASKED]", scrubbed)
    scrubbed = re.sub(r"(?:INR|Rs\.?)\s?[0-9,]+(?:\.[0-9]{2})?", "[INR_AMOUNT_MASKED]", scrubbed)
    return scrubbed


def call_openrouter_extractor(
    case_id: str,
    raw_document_text: str,
    model: str = "openrouter/free",
) -> EvidencePacket:
    """Calls OpenRouter Free-Tier LLM API to extract an EvidencePacket from scrubbed document text."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Run with default '--mode offline' or export OPENROUTER_API_KEY."
        )

    safe_text = scrub_sensitive_financial_pii(raw_document_text)
    # Note: We omit 'response_format': {'type': 'json_object'} because many free-tier
    # providers on OpenRouter filter it out and return 404 No Endpoints Found.
    # Instead, we instruct JSON in the prompt and extract it via regex.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Case ID: {case_id}\n\nDocument Bundle:\n{safe_text}\n\nReturn ONLY valid JSON:"},
        ],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vishvatej/sec43bh-audit-prototype",
            "X-Title": "Sec43Bh-NeuroSymbolic-Audit-Prototype",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            # Robust JSON extraction (handles markdown ```json ... ``` blocks)
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            raw_json = json_match.group(0) if json_match else content
            parsed = json.loads(raw_json)
            return EvidencePacket(**parsed)
    except urllib.error.HTTPError as e:
        server_error = e.read().decode("utf-8", errors="ignore")
        print(f"\n[OpenRouter API Error {e.code}]: {server_error}\n")
        raise


def verify_grounding(raw_document_text: str, packet: EvidencePacket) -> bool:
    """Deterministic anti-hallucination check: verifies verbatim quote exists in raw source text."""
    norm_doc = " ".join(raw_document_text.lower().split())
    norm_quote = " ".join(packet.evidence_quote.lower().split())
    return len(norm_quote) >= 12 and (norm_quote in norm_doc)


# ============================================================================
# 4. STAGE 2: DETERMINISTIC STATUTORY ENGINE (Zero AI / 100% Auditable Math)
# ============================================================================
def parse_iso_date(d_str: Optional[str]) -> Optional[date]:
    if not d_str:
        return None
    return datetime.strptime(d_str, "%Y-%m-%d").date()


def evaluate_section_43bh(
    bundle: SyntheticTestBundle,
    packet: EvidencePacket,
    rules: Dict[str, Any] = STATUTORY_RULES_AY_2026_27,
) -> AuditDecision:
    """Executes exact statutory logic for Section 43B(h) read with Section 15 of MSMED Act."""
    trace: List[str] = []
    fy_end = parse_iso_date(rules["effective_fy_end"])
    pay_date = parse_iso_date(bundle.actual_payment_date)

    def build_decision(
        status: str,
        disallowed: float = 0.0,
        tax_hit: float = 0.0,
        app_start: Optional[str] = None,
        stat_due: Optional[str] = None,
        days_all: Optional[int] = None,
        review_reason: Optional[str] = None,
    ) -> AuditDecision:
        return AuditDecision(
            case_id=bundle.case_id,
            scenario_name=bundle.scenario_name,
            vendor_name=bundle.vendor_name,
            invoice_amount_inr=bundle.invoice_amount_inr,
            status=status,
            expected_ground_truth=bundle.expected_ground_truth,
            matched_ground_truth=(status == bundle.expected_ground_truth),
            disallowed_amount_inr=disallowed,
            cash_tax_impact_inr=tax_hit,
            appointed_day_start=app_start,
            statutory_due_date=stat_due,
            actual_payment_date=bundle.actual_payment_date,
            days_allowed=days_all,
            rule_trace=trace,
            human_review_reason=review_reason,
            rule_version=rules["rule_version"],
        )

    # Gate 1: Substring Grounding & Confidence Threshold Verification
    if not verify_grounding(bundle.raw_document_bundle, packet):
        trace.append("FAIL_GROUNDING: Extracted evidence_quote not found verbatim in source bundle.")
        return build_decision(
            status="ROUTE_TO_HUMAN_CA",
            review_reason="Source quote verification failed (potential OCR/LLM hallucination).",
        )

    if (
        packet.extraction_confidence < rules["confidence_threshold"]
        or packet.expense_nature == "AMBIGUOUS"
        or packet.primary_activity == "COMPOSITE"
    ):
        trace.append(
            f"HOLD_CONFIDENCE: Confidence={packet.extraction_confidence}, "
            f"Expense={packet.expense_nature}, Activity={packet.primary_activity}"
        )
        return build_decision(
            status="ROUTE_TO_HUMAN_CA",
            review_reason="Ambiguous composite contract or low extraction confidence requires CA review.",
        )

    # Gate 2: Statutory Scope Exclusions (Pre-FY Opening Balance, Medium Tier, Trader, CapEx)
    if bundle.is_opening_balance_pre_fy:
        trace.append("EXEMPT_OPENING_BAL: Liability incurred in prior FY; outside current year Sec 43B(h) scope.")
        return build_decision(status="ALLOWABLE")

    if packet.enterprise_tier not in rules["eligible_enterprise_tiers"]:
        trace.append(f"EXEMPT_TIER: Vendor tier is {packet.enterprise_tier}; only MICRO and SMALL covered.")
        return build_decision(status="ALLOWABLE")

    if packet.primary_activity == "TRADING" or packet.nic_code_2digit in rules["excluded_nic_prefixes"]:
        trace.append(
            f"EXEMPT_TRADER: Activity={packet.primary_activity} (NIC {packet.nic_code_2digit}) "
            f"excluded per Ministry of MSME OM dated 02.07.2021."
        )
        return build_decision(status="ALLOWABLE")

    if packet.expense_nature == "CAPEX":
        trace.append("EXEMPT_CAPEX: Capital expenditure is not claimed under Sec 30-37 P&L; Sec 43B(h) inapplicable.")
        return build_decision(status="ALLOWABLE")

    # Gate 3: Paid on or before Fiscal Year-End (March 31)
    if pay_date is not None and pay_date <= fy_end:
        trace.append(
            f"PAID_IN_SAME_FY: Paid on {pay_date.isoformat()} (<= {fy_end.isoformat()}). "
            f"Allowable on actual payment basis in FY 2025-26."
        )
        return build_decision(status="ALLOWABLE", app_start=packet.delivery_date)

    # Gate 4: Determine Base Clock Start (Appointed Day Anchor under Sec 2(b) & Sec 15 MSMED Act)
    inv_date = parse_iso_date(packet.invoice_date)
    deliv_date = parse_iso_date(packet.delivery_date)
    base_start = max(inv_date, deliv_date)
    trace.append(f"BASE_CLOCK: max(invoice={inv_date}, delivery={deliv_date}) -> {base_start}")

    if packet.has_written_quality_objection and packet.objection_date:
        obj_date = parse_iso_date(packet.objection_date)
        days_to_objection = (obj_date - deliv_date).days
        if days_to_objection <= rules["objection_window_days"]:
            if not packet.defect_resolution_date:
                trace.append("OPEN_OBJECTION: Valid quality objection raised within 15 days, resolution date missing.")
                return build_decision(
                    status="ROUTE_TO_HUMAN_CA",
                    review_reason="Valid quality dispute raised within 15 days; verify defect resolution date.",
                )
            res_date = parse_iso_date(packet.defect_resolution_date)
            base_start = res_date
            trace.append(
                f"OBJECTION_RESET: Written quality objection raised in {days_to_objection}d (<=15d). "
                f"Clock reset to defect resolution date {base_start}."
            )
        else:
            trace.append(
                f"OBJECTION_IGNORED: Written objection raised after {days_to_objection}d (>15d statutory limit). "
                f"Original clock retained."
            )

    # Gate 5: Statutory Credit Period Calculation (15 days vs min(Agreed, 45) days)
    if packet.has_bilateral_written_agreement and packet.agreed_credit_days is not None:
        days_allowed = min(packet.agreed_credit_days, rules["max_written_agreement_days"])
        trace.append(
            f"CREDIT_TERMS: Bilateral PO agreement ({packet.agreed_credit_days}d) "
            f"capped at statutory maximum -> {days_allowed} days."
        )
    else:
        days_allowed = rules["default_no_agreement_days"]
        trace.append(f"CREDIT_TERMS: No bilateral written agreement -> default statutory {days_allowed} days.")

    statutory_due = base_start + timedelta(days=days_allowed)
    trace.append(f"STATUTORY_DUE_DATE: {base_start} + {days_allowed} days = {statutory_due}")

    # Gate 6: Heuristic Check for Unrecorded Offline Quality Dispute
    if bundle.has_unexplained_credit_note and (pay_date is None or pay_date > statutory_due):
        trace.append("OFFLINE_DISPUTE_PROXY: Late payment accompanied by credit note adjustment; flagging for CA.")
        return build_decision(
            status="ROUTE_TO_HUMAN_CA",
            app_start=base_start.isoformat(),
            stat_due=statutory_due.isoformat(),
            days_all=days_allowed,
            review_reason="Credit note detected on late invoice; check for unrecorded WhatsApp/gate-pass dispute.",
        )

    # Gate 7: Final Section 43B(h) Disallowance Determination
    if pay_date is not None and pay_date <= statutory_due:
        trace.append(
            f"TIMELY_NEXT_FY_PAYMENT: Paid on {pay_date} which is <= statutory due date {statutory_due}. Allowable."
        )
        return build_decision(
            status="ALLOWABLE",
            app_start=base_start.isoformat(),
            stat_due=statutory_due.isoformat(),
            days_all=days_allowed,
        )

    trace.append(
        f"BREACH_43BH: Unpaid as of {fy_end} AND payment ({bundle.actual_payment_date}) exceeded statutory "
        f"due date ({statutory_due}). First Proviso to Sec 43B (Sec 139(1) ITR due-date relief) is inapplicable."
    )
    tax_hit = round(bundle.invoice_amount_inr * rules["corporate_tax_rate_effective"], 2)
    return build_decision(
        status="DISALLOWED_43BH",
        disallowed=bundle.invoice_amount_inr,
        tax_hit=tax_hit,
        app_start=base_start.isoformat(),
        stat_due=statutory_due.isoformat(),
        days_all=days_allowed,
    )


# ============================================================================
# 5. EMBEDDED 12-CASE SYNTHETIC BENCHMARK SUITE (Public Formats + Realistic Edge Cases)
# ============================================================================
TEST_SUITE: List[SyntheticTestBundle] = [
    SyntheticTestBundle(
        case_id="TC-01",
        scenario_name="Net 60 PO Cap Trap (Capped at 45d)",
        vendor_name="Shree Ganesh Precision Forgings Pvt Ltd",
        invoice_amount_inr=850000.0,
        actual_payment_date="2026-04-02",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="DISALLOWED_43BH",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-MH-19-0041289 | Enterprise Type: SMALL | Major Activity: MANUFACTURING "
            "(NIC 25910: Forging, pressing, stamping of metal).\n"
            "[PURCHASE ORDER #PO-2026-114] Bilateral signed contract between Buyer and Seller: Payment credit "
            "period agreed at Net 60 days from delivery.\n"
            "[TAX INVOICE #INV-882] Dated: 2026-02-01 | Raw Material Forged Steel Flanges (OpEx Consumables) | "
            "Amount: INR 8,50,000.\n"
            "[GOODS RECEIPT NOTE #GRN-901] Goods received and accepted at factory gate on 2026-02-01. No defects."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-01",
            enterprise_tier="SMALL",
            primary_activity="MANUFACTURING",
            nic_code_2digit="25",
            expense_nature="OPEX",
            invoice_date="2026-02-01",
            delivery_date="2026-02-01",
            has_bilateral_written_agreement=True,
            agreed_credit_days=60,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Bilateral signed contract between Buyer and Seller: Payment credit period agreed at Net 60 days",
            extraction_confidence=0.96,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-02",
        scenario_name="Unilateral Invoice Terms (No PO -> 15d Default)",
        vendor_name="Nanded Micro CNC Works",
        invoice_amount_inr=320000.0,
        actual_payment_date="2026-04-05",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="DISALLOWED_43BH",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-MH-15-0019283 | Enterprise Type: MICRO | Major Activity: MANUFACTURING "
            "(NIC 28221: Manufacture of machine tools).\n"
            "[PROCUREMENT LOG] Spot order placed telephonically. No signed Purchase Order or MSA exists.\n"
            "[TAX INVOICE #NMC-409] Dated: 2026-03-10 | Job-work machining of valve housings | INR 3,20,000. "
            "Invoice Footer Note: 'Payment requested within 30 days of bill date.'\n"
            "[GRN #GRN-1120] Received on 2026-03-10."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-02",
            enterprise_tier="MICRO",
            primary_activity="MANUFACTURING",
            nic_code_2digit="28",
            expense_nature="OPEX",
            invoice_date="2026-03-10",
            delivery_date="2026-03-10",
            has_bilateral_written_agreement=False,
            agreed_credit_days=None,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Spot order placed telephonically. No signed Purchase Order or MSA exists.",
            extraction_confidence=0.95,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-03",
        scenario_name="Year-End Straddle Paid Within 45d Window",
        vendor_name="Deccan Cloud Analytics LLP",
        invoice_amount_inr=540000.0,
        actual_payment_date="2026-04-25",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-DL-01-0099120 | Enterprise Type: SMALL | Major Activity: SERVICES "
            "(NIC 62011: Software development and IT consulting).\n"
            "[MASTER SERVICE AGREEMENT] Signed Clause 4.2: Buyer shall remit payment within 45 days of service acceptance.\n"
            "[TAX INVOICE #DCA-991] Dated: 2026-03-20 | Annual Server Log Auditing Services (OpEx) | INR 5,40,000.\n"
            "[SERVICE COMPLETION CERTIFICATE] Signed off on 2026-03-20."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-03",
            enterprise_tier="SMALL",
            primary_activity="SERVICES",
            nic_code_2digit="62",
            expense_nature="OPEX",
            invoice_date="2026-03-20",
            delivery_date="2026-03-20",
            has_bilateral_written_agreement=True,
            agreed_credit_days=45,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Signed Clause 4.2: Buyer shall remit payment within 45 days of service acceptance.",
            extraction_confidence=0.98,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-04",
        scenario_name="Valid Quality Objection (<15d) Resets Clock",
        vendor_name="Sahyadri Copper Windings Co",
        invoice_amount_inr=1200000.0,
        actual_payment_date="2026-04-18",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-MH-26-0055411 | Enterprise Type: MICRO | Major Activity: MANUFACTURING (NIC 27).\n"
            "[SIGNED PO #PO-771] Bilateral credit period: 30 days from delivery/acceptance.\n"
            "[TAX INVOICE #SCW-102] Dated: 2026-02-10 | Copper Stator Coils | Delivered on 2026-02-10.\n"
            "[QUALITY DISPUTE EMAIL LOG] On 2026-02-18 (8 days after delivery), Buyer QA Head sent formal written "
            "rejection email: 'Insulation resistance failed HV testing on 40% of coils; batch quarantined.'\n"
            "[RECTIFICATION CHALLAN] Vendor replaced defective coils and passed QA inspection on 2026-03-22."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-04",
            enterprise_tier="MICRO",
            primary_activity="MANUFACTURING",
            nic_code_2digit="27",
            expense_nature="OPEX",
            invoice_date="2026-02-10",
            delivery_date="2026-02-10",
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=True,
            objection_date="2026-02-18",
            defect_resolution_date="2026-03-22",
            evidence_quote="Vendor replaced defective coils and passed QA inspection on 2026-03-22.",
            extraction_confidence=0.94,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-05",
        scenario_name="Belated Quality Objection (>15d) Ignored",
        vendor_name="Bharat Alloy Castings",
        invoice_amount_inr=675000.0,
        actual_payment_date="2026-04-10",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="DISALLOWED_43BH",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-GJ-01-0033219 | Enterprise Type: SMALL | Major Activity: MANUFACTURING (NIC 24).\n"
            "[SIGNED PO #PO-510] Bilateral credit terms: 30 days.\n"
            "[INVOICE & GRN] Invoice dated 2026-02-01, castings delivered on 2026-02-01.\n"
            "[EMAIL LOG] On 2026-02-22 (21 days after delivery), Buyer emailed complaining of surface blowholes. "
            "Vendor rectified on 2026-03-15."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-05",
            enterprise_tier="SMALL",
            primary_activity="MANUFACTURING",
            nic_code_2digit="24",
            expense_nature="OPEX",
            invoice_date="2026-02-01",
            delivery_date="2026-02-01",
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=True,
            objection_date="2026-02-22",
            defect_resolution_date="2026-03-15",
            evidence_quote="On 2026-02-22 (21 days after delivery), Buyer emailed complaining of surface blowholes.",
            extraction_confidence=0.93,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-06",
        scenario_name="Wholesale Trader with Udyam PDF (NIC 46 Exempt)",
        vendor_name="Metro Industrial Hardware Traders",
        invoice_amount_inr=410000.0,
        actual_payment_date=None,
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-MH-01-0088712 | Enterprise Type: SMALL | Major Activity: TRADING "
            "(NIC 46632: Wholesale of industrial fasteners and hardware, registered under OM dated 02.07.2021).\n"
            "[TAX INVOICE #MIH-77] Dated: 2026-01-15 | Supply of imported SS304 Bolts (OpEx) | Delivered 2026-01-15. "
            "Unpaid as of 2026-03-31."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-06",
            enterprise_tier="SMALL",
            primary_activity="TRADING",
            nic_code_2digit="46",
            expense_nature="OPEX",
            invoice_date="2026-01-15",
            delivery_date="2026-01-15",
            has_bilateral_written_agreement=False,
            agreed_credit_days=None,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="NIC 46632: Wholesale of industrial fasteners and hardware, registered under OM dated 02.07.2021",
            extraction_confidence=0.97,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-07",
        scenario_name="Medium Enterprise Exemption",
        vendor_name="Apex Auto Components Ltd",
        invoice_amount_inr=2450000.0,
        actual_payment_date=None,
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-HR-05-0011223 | Enterprise Type: MEDIUM | Major Activity: MANUFACTURING "
            "(NIC 29301: Manufacture of automotive parts).\n"
            "[TAX INVOICE #AAC-501] Dated: 2026-01-10 | Delivered: 2026-01-10 | Unpaid as of March 31, 2026."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-07",
            enterprise_tier="MEDIUM",
            primary_activity="MANUFACTURING",
            nic_code_2digit="29",
            expense_nature="OPEX",
            invoice_date="2026-01-10",
            delivery_date="2026-01-10",
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Enterprise Type: MEDIUM | Major Activity: MANUFACTURING (NIC 29301",
            extraction_confidence=0.99,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-08",
        scenario_name="Capital Expenditure (CapEx) Fixed Asset Exemption",
        vendor_name="Kulkarni Machine Tools Pvt Ltd",
        invoice_amount_inr=3800000.0,
        actual_payment_date="2026-05-10",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-KA-02-0044918 | Enterprise Type: SMALL | Major Activity: MANUFACTURING (NIC 28).\n"
            "[SIGNED PO #CAPEX-2026-09] Supply and commissioning of 5-Axis CNC Vertical Machining Center "
            "(Capitalized to Plant & Machinery Fixed Asset Block, not debited to P&L).\n"
            "[TAX INVOICE #KMT-88] Dated: 2026-01-20 | Delivered: 2026-01-20 | Paid on 2026-05-10."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-08",
            enterprise_tier="SMALL",
            primary_activity="MANUFACTURING",
            nic_code_2digit="28",
            expense_nature="CAPEX",
            invoice_date="2026-01-20",
            delivery_date="2026-01-20",
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Capitalized to Plant & Machinery Fixed Asset Block, not debited to P&L",
            extraction_confidence=0.96,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-09",
        scenario_name="Overdue In-Year Payment (Paid Before March 31)",
        vendor_name="Vidarbha Enviro Testing Lab",
        invoice_amount_inr=180000.0,
        actual_payment_date="2026-03-28",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-MH-20-0077123 | Enterprise Type: MICRO | Major Activity: SERVICES "
            "(NIC 71200: Technical testing and analysis).\n"
            "[TAX INVOICE #VET-19] Dated: 2026-01-01 | Effluent Water Testing Services (OpEx) | Delivered: 2026-01-01. "
            "No PO exists (15-day statutory limit expired Jan 16). Payment cleared via NEFT on 2026-03-28."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-09",
            enterprise_tier="MICRO",
            primary_activity="SERVICES",
            nic_code_2digit="71",
            expense_nature="OPEX",
            invoice_date="2026-01-01",
            delivery_date="2026-01-01",
            has_bilateral_written_agreement=False,
            agreed_credit_days=None,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Effluent Water Testing Services (OpEx) | Delivered: 2026-01-01.",
            extraction_confidence=0.97,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-10",
        scenario_name="Ambiguous Composite Contract -> Route to Human CA",
        vendor_name="SuryaTej Solar Solutions",
        invoice_amount_inr=950000.0,
        actual_payment_date=None,
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ROUTE_TO_HUMAN_CA",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-RJ-14-0061234 | Enterprise Type: SMALL | Lists both NIC 46 (Trading) "
            "and NIC 43 (Electrical installation services).\n"
            "[TAX INVOICE #STS-304] Dated: 2026-02-05 | Lump-sum composite bill for traded third-party solar "
            "panels (70%) and rooftop maintenance service (30%) without separate line-item valuation."
        ),
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-10",
            enterprise_tier="SMALL",
            primary_activity="COMPOSITE",
            nic_code_2digit="46",
            expense_nature="AMBIGUOUS",
            invoice_date="2026-02-05",
            delivery_date="2026-02-05",
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Lump-sum composite bill for traded third-party solar panels (70%) and rooftop maintenance",
            extraction_confidence=0.72,
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-11",
        scenario_name="[FAILURE MODE 1] DD/MM vs MM/DD Challan Date Ambiguity",
        vendor_name="Godavari Fasteners & Springs",
        invoice_amount_inr=490000.0,
        actual_payment_date="2026-04-01",
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="DISALLOWED_43BH",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-MH-19-0022910 | Enterprise Type: MICRO | Major Activity: MANUFACTURING (NIC 25).\n"
            "[SIGNED PO #PO-812] Bilateral credit period: 30 days from delivery.\n"
            "[DELIVERY CHALLAN #DC-44] Rubber stamp reads: 'Material Received at Gate on 03/02/2026' "
            "(Indian DD/MM/YYYY format = February 3, 2026, making 30-day due date March 5, 2026).\n"
            "[TAX INVOICE #GFS-90] Accounting team generated invoice late on 02-Mar-2026."
        ),
        # Honest Failure: Stage 1 LLM sees '02-Mar-2026' on the invoice and misparses '03/02/2026' as US MM/DD/YYYY
        # (2026-03-02 instead of 2026-02-03), shifting the 30-day window to 2026-04-01 and causing a False Negative!
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-11",
            enterprise_tier="MICRO",
            primary_activity="MANUFACTURING",
            nic_code_2digit="25",
            expense_nature="OPEX",
            invoice_date="2026-03-02",
            delivery_date="2026-03-02",  # LLM hallucinated MM/DD/YYYY (March 2) instead of DD/MM/YYYY (Feb 3)
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Material Received at Gate on 03/02/2026",
            extraction_confidence=0.91,
        ),
        failure_analysis_note=(
            "HONEST FAILURE 1 (False Negative): Challan stamp read '03/02/2026' (DD/MM/YYYY = Feb 3, 2026), "
            "but because the accounting invoice was dated '02-Mar-2026', the LLM normalized '03/02/2026' using "
            "US MM/DD/YYYY format ('2026-03-02'). This pushed the 30-day statutory due date from March 5 to April 1, "
            "erroneously marking a disallowable invoice as ALLOWABLE."
        ),
    ),
    SyntheticTestBundle(
        case_id="TC-12",
        scenario_name="[FAILURE MODE 2] Dual-NIC Udyam PDF + Opaque SKU",
        vendor_name="Precision Flow Control Systems",
        invoice_amount_inr=730000.0,
        actual_payment_date=None,
        is_opening_balance_pre_fy=False,
        has_unexplained_credit_note=False,
        expected_ground_truth="ALLOWABLE",
        raw_document_bundle=(
            "[UDYAM CERTIFICATE] UDYAM-GJ-06-0091827 | Enterprise Type: SMALL | Page 1 Primary Activity: "
            "MANUFACTURING (NIC 25). Page 2 Additional Activity: WHOLESALE TRADE (NIC 46).\n"
            "[TAX INVOICE #PFC-211] Dated: 2026-01-25 | Supply of High-Pressure Ball Valves (OpEx Spares) | "
            "Delivered: 2026-01-25. (Note: Vendor actually traded imported Korean valves on this bill, which is "
            "exempt from Sec 43B(h), but invoice line item does not explicitly mention 'Traded Goods')."
        ),
        # Honest Failure: Stage 1 LLM locks onto Page 1 NIC 25 (MANUFACTURING) with 0.90 confidence instead of
        # flagging the dual-NIC ambiguity, triggering a False Positive disallowance.
        llm_extracted_fixture=EvidencePacket(
            case_id="TC-12",
            enterprise_tier="SMALL",
            primary_activity="MANUFACTURING",  # LLM missed Page 2 NIC 46 Trading activity overlap
            nic_code_2digit="25",
            expense_nature="OPEX",
            invoice_date="2026-01-25",
            delivery_date="2026-01-25",
            has_bilateral_written_agreement=True,
            agreed_credit_days=30,
            has_written_quality_objection=False,
            objection_date=None,
            defect_resolution_date=None,
            evidence_quote="Page 1 Primary Activity: MANUFACTURING (NIC 25).",
            extraction_confidence=0.90,
        ),
        failure_analysis_note=(
            "HONEST FAILURE 2 (False Positive): Vendor's Udyam PDF listed both NIC 25 (Manufacturing) on Page 1 "
            "and NIC 46 (Wholesale Trade) on Page 2. Because the invoice merely said 'Supply of High-Pressure Ball "
            "Valves', the LLM defaulted to the primary NIC 25 Manufacturing tag with 0.90 confidence, causing "
            "Stage 2 to disallow INR 7,30,000 on an exempt trading transaction."
        ),
    ),
]


# ============================================================================
# 6. CLI RUNNER & BENCHMARK REPORTER
# ============================================================================
def run_benchmark(mode: str = "offline", model: str = "openrouter/free") -> List[AuditDecision]:
    decisions: List[AuditDecision] = []
    for bundle in TEST_SUITE:
        if mode == "live":
            print(f"[LIVE OPENROUTER] Extracting {bundle.case_id} ({bundle.vendor_name}) via {model}...")
            packet = call_openrouter_extractor(bundle.case_id, bundle.raw_document_bundle, model=model)
        else:
            packet = bundle.llm_extracted_fixture

        decision = evaluate_section_43bh(bundle, packet)
        decisions.append(decision)
    return decisions


def print_report(decisions: List[AuditDecision]) -> None:
    print("=" * 122)
    print("TRESATA TAKE-HOME PROTOTYPE: SECTION 43B(h) NEURO-SYMBOLIC AUDIT ENGINE BENCHMARK (12 SYNTHETIC CASES)")
    print(f"Active Statutory Rule Version: {STATUTORY_RULES_AY_2026_27['rule_version']} | Effective FY: 2025-26 (AY 2026-27)")
    print("=" * 122)
    header = (
        f"{'ID':<6} | {'Scenario':<42} | {'Due Date':<10} | {'Paid Date':<10} | "
        f"{'System Output':<18} | {'Ground Truth':<18} | {'Match'}"
    )
    print(header)
    print("-" * 122)

    passed = 0
    total_disallowed = 0.0
    total_tax_impact = 0.0

    for d in decisions:
        if d.matched_ground_truth:
            passed += 1
        if d.status == "DISALLOWED_43BH":
            total_disallowed += d.disallowed_amount_inr
            total_tax_impact += d.cash_tax_impact_inr

        due_str = d.statutory_due_date or "N/A"
        paid_str = d.actual_payment_date or "UNPAID"
        match_str = "PASS" if d.matched_ground_truth else "FAIL (!)"
        print(
            f"{d.case_id:<6} | {d.scenario_name[:42]:<42} | {due_str:<10} | {paid_str:<10} | "
            f"{d.status:<18} | {d.expected_ground_truth:<18} | {match_str}"
        )

    print("-" * 122)
    print(
        f"BENCHMARK ACCURACY: {passed}/{len(decisions)} ({passed / len(decisions) * 100:.1f}%) | "
        f"Total Flagged Disallowance: INR {total_disallowed:,.2f} | "
        f"Estimated Cash Tax Impact (@25.168%): INR {total_tax_impact:,.2f}"
    )
    print("=" * 122)

    print("\nDETAILED AUDIT FAILURE REPORT (HONEST REPORTING OF WHERE THE PROTOTYPE FAILS):")
    print("-" * 122)
    for bundle, d in zip(TEST_SUITE, decisions):
        if not d.matched_ground_truth and bundle.failure_analysis_note:
            print(f"* {bundle.case_id} ({bundle.vendor_name}):")
            print(f"  - Expected: {d.expected_ground_truth} | Actual Output: {d.status}")
            print(f"  - Root Cause: {bundle.failure_analysis_note}")
            print(f"  - Stage 2 Trace: {' -> '.join(d.rule_trace)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Section 43B(h) Hybrid Neuro-Symbolic Statutory Audit Prototype"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Run using embedded LLM extraction fixtures ('offline') or live OpenRouter API ('live').",
    )
    parser.add_argument(
        "--model",
        default="openrouter/free",
        help="OpenRouter model ID when running in --mode live.",
    )
    parser.add_argument(
        "--export-json",
        default=None,
        help="Optional filepath to export full Form 3CD Clause 22 audit working paper JSON.",
    )
    args = parser.parse_args()

    decisions = run_benchmark(mode=args.mode, model=args.model)
    print_report(decisions)

    if args.export_json:
        output_payload = {
            "rule_metadata": STATUTORY_RULES_AY_2026_27,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_decisions": [asdict(d) for d in decisions],
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"Exported full audit trace JSON to: {args.export_json}")


if __name__ == "__main__":
    main()
