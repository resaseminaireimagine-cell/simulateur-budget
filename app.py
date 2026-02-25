"""
App: Salaire nécessaire pour soutenir un train de vie (France, salarié)
- Dépenses éditables (fréquences + mensualisation)
- Emprunts multi-prêts (mensualités + assurance + frais)
- Hypothèses (PAS, ratio net->brut, buffer, épargne)
- Scénarios (Minimum / Confort / Ambitieux)
- Exports CSV + "sauvegarde simple" (fichier) + PDF optionnel
"""

from __future__ import annotations

import io
import json
import math
from datetime import date, datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


APP_TITLE = "Salaire requis — Train de vie (France, salarié)"
CURRENCY = "€"

FREQ_OPTIONS = ["mensuel", "hebdo", "trimestriel", "annuel", "ponctuel"]
LEVEL_OPTIONS = ["Essentiel", "Confort"]
LOAN_TYPE_OPTIONS = ["immo", "auto", "conso", "etudiant", "autre"]
INS_TYPE_OPTIONS = ["aucune", "taux_annuel_sur_capital", "mensuel"]
BUFFER_BASE_OPTIONS = ["depenses_hors_prets", "depenses_plus_prets", "revenu_net_apres_ir"]


# -----------------------
# Finance helpers
# -----------------------
def pmt(rate: float, nper: int, pv: float) -> float:
    """Mensualité (positive) pour un prêt pv (positif), taux périodique rate."""
    if nper <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return pv / nper
    return pv * (rate / (1 - (1 + rate) ** (-nper)))


def loan_payment(capital: float, annual_rate: float, years: float) -> float:
    """Mensualité hors assurance et hors frais étalés."""
    n = int(round(years * 12))
    r = (annual_rate / 100.0) / 12.0
    return float(pmt(r, n, capital))


def amortization_schedule(capital: float, annual_rate: float, years: float) -> pd.DataFrame:
    """Tableau d'amortissement (hors assurance/frais)."""
    n = int(round(years * 12))
    r = (annual_rate / 100.0) / 12.0
    payment = loan_payment(capital, annual_rate, years)

    balance = capital
    rows = []
    for k in range(1, n + 1):
        interest = balance * r if r > 0 else 0.0
        principal = payment - interest
        if principal > balance:
            principal = balance
            payment_eff = principal + interest
        else:
            payment_eff = payment
        balance = max(0.0, balance - principal)
        rows.append(
            {
                "Mois": k,
                "Mensualité": payment_eff,
                "Intérêts": interest,
                "Principal": principal,
                "Capital restant dû": balance,
            }
        )
        if balance <= 1e-6:
            break
    return pd.DataFrame(rows)


# -----------------------
# Monthlyization expenses
# -----------------------
def months_between_inclusive(d1: date, d2: date) -> int:
    if d2 < d1:
        return 0
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + 1


def monthlyize_expenses(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = df.copy()

    monthly_vals = []
    annual_vals = []

    for _, row in df.iterrows():
        amount = float(row.get("montant", 0) or 0)
        freq = str(row.get("frequence", "mensuel") or "mensuel").strip().lower()
        start = row.get("date_debut", None)
        end = row.get("date_fin", None)

        if amount < 0:
            warnings.append(f"Dépense '{row.get('nom','(sans nom)')}' : montant négatif.")
        if freq not in FREQ_OPTIONS:
            warnings.append(f"Dépense '{row.get('nom','(sans nom)')}' : fréquence inconnue '{freq}' (traitée comme mensuel).")
            freq = "mensuel"

        if freq == "mensuel":
            m = amount
        elif freq == "hebdo":
            m = amount * (52 / 12)
        elif freq == "trimestriel":
            m = amount / 3
        elif freq == "annuel":
            m = amount / 12
        elif freq == "ponctuel":
            if isinstance(start, (datetime, date)):
                s = start.date() if isinstance(start, datetime) else start
            else:
                s = None
            if isinstance(end, (datetime, date)):
                e = end.date() if isinstance(end, datetime) else end
            else:
                e = None

            if s is None:
                warnings.append(f"Dépense ponctuelle '{row.get('nom','(sans nom)')}' : date_debut manquante → étalée sur 12 mois.")
                months = 12
            else:
                if e is None:
                    e = s
                months = months_between_inclusive(s, e)
                if months <= 0:
                    warnings.append(f"Dépense ponctuelle '{row.get('nom','(sans nom)')}' : dates incohérentes → étalée sur 12 mois.")
                    months = 12
            m = amount / months
        else:
            m = amount

        monthly_vals.append(m)
        annual_vals.append(m * 12)

    df["monthly_equiv"] = monthly_vals
    df["annual_equiv"] = annual_vals
    return df, warnings


# -----------------------
# Validation
# -----------------------
def validate_inputs(expenses_df: pd.DataFrame, loans_df: pd.DataFrame, assumptions: Dict) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    pas = float(assumptions.get("pas_rate", 0.081))
    ratio = float(assumptions.get("ratio_net_brut", 0.78))
    if not (0 <= pas < 1):
        errors.append("PAS : doit être compris entre 0% et 99,99%.")
    if not (0 < ratio <= 1.0):
        errors.append("Ratio net→brut : doit être > 0 et ≤ 1 (ex: 0,78).")

    buffer_pct = float(assumptions.get("buffer_pct", 0.10))
    if buffer_pct < 0 or buffer_pct > 2:
        warnings.append("Buffer : valeur étrange (attendu ~0% à 50%).")

    sav_pct = float(assumptions.get("savings_pct_income", 0.0))
    if sav_pct < 0 or sav_pct > 0.8:
        warnings.append("Épargne % : valeur étrange (attendu ~0% à 30%).")

    if expenses_df is None or expenses_df.empty:
        warnings.append("Aucune dépense : les résultats seront sous-estimés.")

    if loans_df is not None and not loans_df.empty:
        for _, r in loans_df.iterrows():
            name = str(r.get("nom", "(sans nom)"))
            capital = float(r.get("capital", 0) or 0)
            years = float(r.get("duree_annees", 0) or 0)
            rate = float(r.get("taux_annuel_pct", 0) or 0)
            if capital < 0:
                errors.append(f"Prêt '{name}' : capital négatif.")
            if years <= 0:
                errors.append(f"Prêt '{name}' : durée doit être > 0.")
            if rate < 0:
                errors.append(f"Prêt '{name}' : taux annuel négatif.")

    return errors, warnings


# -----------------------
# Core computations
# -----------------------
def compute_net_after_ir(
    expenses_monthly: float,
    loans_monthly: float,
    savings_fixed: float,
    savings_pct_income: float,
    buffer_amount: float,
    buffer_pct_income: float,
) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Si épargne% ou buffer% sur revenu:
      need = base + (savings_pct_income + buffer_pct_income) * need
      => need = base / (1 - k)
    """
    notes: List[str] = []
    base = expenses_monthly + loans_monthly + savings_fixed + buffer_amount

    k = savings_pct_income + buffer_pct_income
    if k >= 0.95:
        return float("nan"), {}, ["% épargne + % buffer sur revenu trop élevés (≥95%)."]

    if k > 0:
        need = base / (1 - k)
        notes.append("Besoin net après IR résolu avec % sur revenu (épargne/buffer).")
    else:
        need = base

    breakdown = {
        "depenses": expenses_monthly,
        "prets": loans_monthly,
        "epargne_fixe": savings_fixed,
        "buffer": buffer_amount,
        "epargne_pct": savings_pct_income * need if savings_pct_income > 0 else 0.0,
        "buffer_pct_revenu": buffer_pct_income * need if buffer_pct_income > 0 else 0.0,
    }
    return need, breakdown, notes


def compute_net_before_ir(net_after_ir: float, pas_rate: float) -> float:
    return net_after_ir / (1 - pas_rate) if pas_rate < 1 else float("nan")


def net_to_gross(net_before_ir: float, ratio_net_brut: float) -> float:
    return net_before_ir / ratio_net_brut if ratio_net_brut > 0 else float("nan")


def compute_loans(loans_df: pd.DataFrame) -> Tuple[pd.DataFrame, float, List[str]]:
    if loans_df is None or loans_df.empty:
        return pd.DataFrame(), 0.0, []

    warnings: List[str] = []
    out = loans_df.copy()

    payments, ins_monthly, fees_monthly, total_monthly, interest_cost = [], [], [], [], []

    for _, r in out.iterrows():
        name = str(r.get("nom", "(sans nom)"))
        capital = float(r.get("capital", 0) or 0)
        rate = float(r.get("taux_annuel_pct", 0) or 0)
        years = float(r.get("duree_annees", 0) or 0)
        n = int(round(years * 12)) if years > 0 else 0

        pay = loan_payment(capital, rate, years) if capital > 0 and years > 0 else 0.0

        ins_type = str(r.get("assurance_type", "aucune"))
        if ins_type == "taux_annuel_sur_capital":
            ins_rate = float(r.get("assurance_taux_annuel_pct", 0) or 0) / 100.0
            ins = capital * ins_rate / 12.0
        elif ins_type == "mensuel":
            ins = float(r.get("assurance_mensuelle", 0) or 0)
        else:
            ins = 0.0

        fees = float(r.get("frais", 0) or 0)
        amortize = bool(r.get("frais_etales_mensuel", True))
        f_m = (fees / n) if (amortize and n > 0) else 0.0

        tot = pay + ins + f_m

        payments.append(pay)
        ins_monthly.append(ins)
        fees_monthly.append(f_m)
        total_monthly.append(tot)

        interest = (pay * n - capital) if n > 0 else 0.0
        if interest < -1e-6:
            warnings.append(f"Prêt '{name}': intérêts calculés négatifs (check inputs).")
        interest_cost.append(max(0.0, interest))

    out["mensualite_hors_assurance"] = payments
    out["assurance_mensuelle_calc"] = ins_monthly
    out["frais_mensuels_calc"] = fees_monthly
    out["mensualite_totale_calc"] = total_monthly
    out["cout_interets_estime"] = interest_cost

    total = float(np.nansum(out["mensualite_totale_calc"].values))
    return out, total, warnings


def compute_buffer_amount(buffer_pct: float, buffer_base: str, expenses_monthly: float, loans_monthly: float) -> float:
    if buffer_pct <= 0:
        return 0.0
    if buffer_base == "depenses_plus_prets":
        return buffer_pct * (expenses_monthly + loans_monthly)
    if buffer_base == "revenu_net_apres_ir":
        return 0.0  # géré via % revenu si choisi
    return buffer_pct * expenses_monthly  # défaut


def compute_scenarios(expenses_df_m: pd.DataFrame, loans_total_monthly: float, assumptions: Dict) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    pas = float(assumptions["pas_rate"])
    ratio = float(assumptions["ratio_net_brut"])
    show_employer = bool(assumptions.get("show_employer_cost", False))
    employer_coef = float(assumptions.get("employer_cost_coef", 1.45))

    buffer_pct = float(assumptions.get("buffer_pct", 0.10))
    buffer_base = str(assumptions.get("buffer_base", "depenses_hors_prets"))

    savings_fixed_base = float(assumptions.get("savings_fixed", 300.0))
    savings_pct_base = float(assumptions.get("savings_pct_income", 0.0))
    buffer_pct_income_base = float(assumptions.get("buffer_pct_income_base", 0.0))

    # Minimum
    savings_fixed_min = float(assumptions.get("savings_fixed_min", savings_fixed_base))
    savings_pct_min = float(assumptions.get("savings_pct_min", savings_pct_base))
    buffer_pct_min = float(assumptions.get("buffer_pct_min", buffer_pct))
    buffer_pct_income_min = float(assumptions.get("buffer_pct_income_min", 0.0))

    # Ambitieux boosts
    amb_savings_fixed_boost = float(assumptions.get("amb_savings_fixed_boost", 200.0))
    amb_savings_pct_boost = float(assumptions.get("amb_savings_pct_boost", 0.0))
    amb_buffer_pct_boost = float(assumptions.get("amb_buffer_pct_boost", 0.0))
    amb_buffer_pct_income_boost = float(assumptions.get("amb_buffer_pct_income_boost", 0.0))

    active = expenses_df_m[expenses_df_m["actif"].fillna(True) == True].copy()
    ess = active[active["niveau"].fillna("Essentiel") == "Essentiel"]
    conf = active

    ess_m = float(np.nansum(ess["monthly_equiv"].values))
    conf_m = float(np.nansum(conf["monthly_equiv"].values))

    scenario_defs = [
        ("Minimum", ess_m, savings_fixed_min, savings_pct_min, buffer_pct_min, buffer_pct_income_min),
        ("Confort", conf_m, savings_fixed_base, savings_pct_base, buffer_pct, buffer_pct_income_base),
        ("Ambitieux", conf_m, savings_fixed_base + amb_savings_fixed_boost, savings_pct_base + amb_savings_pct_boost,
         buffer_pct + amb_buffer_pct_boost, buffer_pct_income_base + amb_buffer_pct_income_boost),
    ]

    rows = []
    details: Dict[str, Dict] = {}

    for name, exp_m, sav_fixed, sav_pct, b_pct, b_pct_income in scenario_defs:
        buffer_amount = compute_buffer_amount(b_pct, buffer_base, exp_m, loans_total_monthly)

        need_net_after, breakdown, notes = compute_net_after_ir(
            expenses_monthly=exp_m,
            loans_monthly=loans_total_monthly,
            savings_fixed=sav_fixed,
            savings_pct_income=sav_pct,
            buffer_amount=buffer_amount,
            buffer_pct_income=b_pct_income if buffer_base == "revenu_net_apres_ir" else 0.0,
        )

        net_before = compute_net_before_ir(need_net_after, pas)
        gross = net_to_gross(net_before, ratio)
        employer_cost = gross * employer_coef if show_employer else np.nan

        rows.append(
            {
                "Scénario": name,
                "Dépenses mensualisées": exp_m,
                "Mensualités prêts": loans_total_monthly,
                "Buffer (montant)": buffer_amount + (breakdown.get("buffer_pct_revenu", 0.0) if breakdown else 0.0),
                "Épargne (fixe)": sav_fixed,
                "Épargne (% revenu)": breakdown.get("epargne_pct", 0.0) if breakdown else 0.0,
                "Net après IR requis": need_net_after,
                "Net avant IR requis": net_before,
                "Brut estimé requis": gross,
                "Coût employeur estimé": employer_cost,
                "Annuel (net après IR)": need_net_after * 12 if not math.isnan(need_net_after) else np.nan,
                "Annuel (brut)": gross * 12 if not math.isnan(gross) else np.nan,
            }
        )
        details[name] = {"breakdown": breakdown, "notes": notes}

    return pd.DataFrame(rows), details


# -----------------------
# Defaults (example data)
# -----------------------
def default_expenses() -> pd.DataFrame:
    data = [
        {"nom": "Loyer", "categorie": "Logement", "montant": 1500.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Charges", "categorie": "Logement", "montant": 150.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Électricité/Gaz", "categorie": "Logement", "montant": 120.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Internet", "categorie": "Logement", "montant": 40.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Assurance habitation", "categorie": "Assurances", "montant": 18.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},

        {"nom": "Courses", "categorie": "Alimentation", "montant": 450.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Restaurants", "categorie": "Alimentation", "montant": 220.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Confort", "commentaire": "", "actif": True},

        {"nom": "Navigo", "categorie": "Transport", "montant": 86.4, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Essentiel", "commentaire": "", "actif": True},

        {"nom": "Salle de sport", "categorie": "Abonnements", "montant": 45.0, "frequence": "mensuel", "date_debut": None, "date_fin": None,
         "niveau": "Confort", "commentaire": "", "actif": True},

        {"nom": "Vacances (mensualisées)", "categorie": "Loisirs", "montant": 1800.0, "frequence": "annuel", "date_debut": None, "date_fin": None,
         "niveau": "Confort", "commentaire": "", "actif": True},
    ]
    return pd.DataFrame(data)


def default_loans() -> pd.DataFrame:
    data = [
        {
            "nom": "Prêt auto (exemple)",
            "type": "auto",
            "capital": 15000.0,
            "taux_annuel_pct": 4.2,
            "duree_annees": 5.0,
            "date_depart": None,
            "assurance_type": "mensuel",
            "assurance_taux_annuel_pct": 0.0,
            "assurance_mensuelle": 20.0,
            "frais": 300.0,
            "frais_etales_mensuel": True,
        }
    ]
    return pd.DataFrame(data)


def default_assumptions() -> Dict:
    return {
        "pas_rate": 0.081,            # 8,1%
        "ratio_net_brut": 0.78,
        "show_employer_cost": False,
        "employer_cost_coef": 1.45,

        "buffer_pct": 0.10,
        "buffer_base": "depenses_hors_prets",

        "savings_fixed": 300.0,
        "savings_pct_income": 0.0,
        "buffer_pct_income_base": 0.0,

        "savings_fixed_min": 300.0,
        "savings_pct_min": 0.0,
        "buffer_pct_min": 0.10,
        "buffer_pct_income_min": 0.0,

        "amb_savings_fixed_boost": 200.0,
        "amb_savings_pct_boost": 0.0,
        "amb_buffer_pct_boost": 0.0,
        "amb_buffer_pct_income_boost": 0.0,
    }


# -----------------------
# Save/Load bundle (simple file)
# -----------------------
def bundle_to_json(expenses_df: pd.DataFrame, loans_df: pd.DataFrame, assumptions: Dict) -> str:
    payload = {
        "version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "assumptions": assumptions,
        "expenses": expenses_df.to_dict(orient="records"),
        "loans": loans_df.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def json_to_bundle(s: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    obj = json.loads(s)
    exp = pd.DataFrame(obj.get("expenses", []))
    loans = pd.DataFrame(obj.get("loans", []))
    assumptions = obj.get("assumptions", {})
    return exp, loans, assumptions


def make_pdf_summary(scenarios_df: pd.DataFrame, assumptions: Dict) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def draw_line(y, text, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(40, y, text)

    y = h - 50
    draw_line(y, "Résumé — Salaire requis (train de vie)", bold=True)
    y -= 20
    draw_line(y, f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 20

    pas = assumptions.get("pas_rate", 0.081)
    ratio = assumptions.get("ratio_net_brut", 0.78)
    draw_line(y, f"Hypothèses: PAS={pas*100:.2f}% | ratio net→brut={ratio:.2f}")
    y -= 25

    cols = ["Scénario", "Net après IR requis", "Net avant IR requis", "Brut estimé requis"]
    df = scenarios_df[cols].copy()

    draw_line(y, "Scénarios (mensuel):", bold=True)
    y -= 15
    for _, r in df.iterrows():
        if y < 80:
            c.showPage()
            y = h - 50
        draw_line(
            y,
            f"- {r['Scénario']}: Net après IR={r['Net après IR requis']:.0f}{CURRENCY} | "
            f"Net avant IR={r['Net avant IR requis']:.0f}{CURRENCY} | Brut={r['Brut estimé requis']:.0f}{CURRENCY}"
        )
        y -= 14

    c.showPage()
    c.save()
    return buf.getvalue()


# -----------------------
# UI state
# -----------------------
def init_state():
    if "expenses_df" not in st.session_state:
        st.session_state.expenses_df = default_expenses()
    if "loans_df" not in st.session_state:
        st.session_state.loans_df = default_loans()
    if "assumptions" not in st.session_state:
        st.session_state.assumptions = default_assumptions()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Besoin cash (net après IR) → net avant IR → brut (approx). Tout est modifiable, exportable, et simple à reprendre.")

    tab_exp, tab_loans, tab_hyp, tab_results, tab_export = st.tabs(
        ["Dépenses", "Emprunts", "Hypothèses", "Résultats & scénarios", "Exports & sauvegarde"]
    )

    with tab_exp:
        st.subheader("Dépenses (éditables)")
        st.write("Mets **Essentiel/Confort** proprement — c’est la base des scénarios.")
        exp_editor = st.data_editor(
            st.session_state.expenses_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nom": st.column_config.TextColumn("Nom", required=True),
                "categorie": st.column_config.TextColumn("Catégorie", required=True),
                "montant": st.column_config.NumberColumn(f"Montant ({CURRENCY})", min_value=0.0, step=10.0, format="%.2f"),
                "frequence": st.column_config.SelectboxColumn("Fréquence", options=FREQ_OPTIONS, required=True),
                "date_debut": st.column_config.DateColumn("Date début (ponctuel)", format="DD/MM/YYYY"),
                "date_fin": st.column_config.DateColumn("Date fin (ponctuel)", format="DD/MM/YYYY"),
                "niveau": st.column_config.SelectboxColumn("Niveau", options=LEVEL_OPTIONS, required=True),
                "commentaire": st.column_config.TextColumn("Commentaire"),
                "actif": st.column_config.CheckboxColumn("Actif"),
            },
        )
        st.session_state.expenses_df = exp_editor

        exp_m_df, exp_warn = monthlyize_expenses(st.session_state.expenses_df)
        st.markdown("#### Équivalents calculés")
        st.dataframe(
            exp_m_df[["nom", "categorie", "niveau", "frequence", "montant", "monthly_equiv", "annual_equiv", "actif"]],
            use_container_width=True,
        )
        if exp_warn:
            st.warning("Avertissements dépenses:\n- " + "\n- ".join(exp_warn))

    with tab_loans:
        st.subheader("Emprunts / dettes (multi-prêts)")
        loans_editor = st.data_editor(
            st.session_state.loans_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nom": st.column_config.TextColumn("Nom", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=LOAN_TYPE_OPTIONS, required=True),
                "capital": st.column_config.NumberColumn(f"Capital ({CURRENCY})", min_value=0.0, step=100.0, format="%.2f"),
                "taux_annuel_pct": st.column_config.NumberColumn("Taux annuel (%)", min_value=0.0, step=0.1, format="%.2f"),
                "duree_annees": st.column_config.NumberColumn("Durée (années)", min_value=0.1, step=0.5, format="%.1f"),
                "date_depart": st.column_config.DateColumn("Date départ (optionnel)", format="DD/MM/YYYY"),
                "assurance_type": st.column_config.SelectboxColumn("Assurance type", options=INS_TYPE_OPTIONS, required=True),
                "assurance_taux_annuel_pct": st.column_config.NumberColumn("Assurance taux annuel (%)", min_value=0.0, step=0.05, format="%.2f"),
                "assurance_mensuelle": st.column_config.NumberColumn(f"Assurance mensuelle ({CURRENCY})", min_value=0.0, step=1.0, format="%.2f"),
                "frais": st.column_config.NumberColumn(f"Frais ({CURRENCY})", min_value=0.0, step=10.0, format="%.2f"),
                "frais_etales_mensuel": st.column_config.CheckboxColumn("Frais étalés mensuellement"),
            },
        )
        st.session_state.loans_df = loans_editor

        loans_calc_df, loans_total_monthly, loan_warn = compute_loans(st.session_state.loans_df)
        st.markdown("#### Mensualités calculées")
        if loans_calc_df.empty:
            st.info("Aucun prêt.")
        else:
            st.dataframe(
                loans_calc_df[
                    [
                        "nom",
                        "type",
                        "capital",
                        "taux_annuel_pct",
                        "duree_annees",
                        "mensualite_hors_assurance",
                        "assurance_mensuelle_calc",
                        "frais_mensuels_calc",
                        "mensualite_totale_calc",
                        "cout_interets_estime",
                    ]
                ],
                use_container_width=True,
            )
            st.metric("Total mensualités prêts", f"{loans_total_monthly:,.2f} {CURRENCY}".replace(",", " "))

        if loan_warn:
            st.warning("Avertissements prêts:\n- " + "\n- ".join(loan_warn))

        with st.expander("Option: tableau d’amortissement (1 prêt)"):
            if not loans_calc_df.empty:
                pick = st.selectbox("Choisir un prêt", loans_calc_df["nom"].tolist())
                row = loans_calc_df[loans_calc_df["nom"] == pick].iloc[0]
                st.dataframe(
                    amortization_schedule(float(row["capital"]), float(row["taux_annuel_pct"]), float(row["duree_annees"])),
                    use_container_width=True,
                )

    with tab_hyp:
        st.subheader("Hypothèses (modifiables)")
        a = st.session_state.assumptions

        c1, c2, c3 = st.columns(3)
        with c1:
            a["pas_rate"] = st.number_input("Taux PAS (0.081 = 8,1%)", 0.0, 0.99, float(a.get("pas_rate", 0.081)), 0.001, format="%.3f")
        with c2:
            a["ratio_net_brut"] = st.number_input("Ratio net avant IR → brut", 0.30, 0.95, float(a.get("ratio_net_brut", 0.78)), 0.01, format="%.2f")
        with c3:
            a["show_employer_cost"] = st.checkbox("Afficher coût employeur", value=bool(a.get("show_employer_cost", False)))
            a["employer_cost_coef"] = st.number_input("Coef coût employeur", 1.0, 3.0, float(a.get("employer_cost_coef", 1.45)), 0.05, format="%.2f")

        st.divider()

        d1, d2, d3 = st.columns(3)
        with d1:
            a["buffer_pct"] = st.number_input("Buffer imprévus (0.10 = 10%)", 0.0, 1.0, float(a.get("buffer_pct", 0.10)), 0.01, format="%.2f")
            a["buffer_base"] = st.selectbox("Base buffer", BUFFER_BASE_OPTIONS, index=BUFFER_BASE_OPTIONS.index(str(a.get("buffer_base", "depenses_hors_prets"))))
        with d2:
            a["savings_fixed"] = st.number_input(f"Épargne fixe ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("savings_fixed", 300.0)), 50.0, format="%.0f")
        with d3:
            a["savings_pct_income"] = st.number_input("Épargne % du net après IR", 0.0, 0.8, float(a.get("savings_pct_income", 0.0)), 0.01, format="%.2f")
            a["buffer_pct_income_base"] = st.number_input("Buffer % du revenu net après IR (option)", 0.0, 0.8, float(a.get("buffer_pct_income_base", 0.0)), 0.01, format="%.2f")

        st.divider()
        st.markdown("### Scénarios")

        s1, s2, s3 = st.columns(3)
        with s1:
            a["savings_fixed_min"] = st.number_input(f"[Minimum] Épargne fixe ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("savings_fixed_min", 300.0)), 50.0, format="%.0f")
            a["savings_pct_min"] = st.number_input("[Minimum] Épargne % revenu", 0.0, 0.8, float(a.get("savings_pct_min", 0.0)), 0.01, format="%.2f")
        with s2:
            a["buffer_pct_min"] = st.number_input("[Minimum] Buffer %", 0.0, 1.0, float(a.get("buffer_pct_min", 0.10)), 0.01, format="%.2f")
            a["buffer_pct_income_min"] = st.number_input("[Minimum] Buffer % revenu", 0.0, 0.8, float(a.get("buffer_pct_income_min", 0.0)), 0.01, format="%.2f")
        with s3:
            a["amb_savings_fixed_boost"] = st.number_input(f"[Ambitieux] Bonus épargne fixe ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("amb_savings_fixed_boost", 200.0)), 50.0, format="%.0f")
            a["amb_savings_pct_boost"] = st.number_input("[Ambitieux] Bonus épargne %", 0.0, 0.8, float(a.get("amb_savings_pct_boost", 0.0)), 0.01, format="%.2f")
            a["amb_buffer_pct_boost"] = st.number_input("[Ambitieux] Bonus buffer %", 0.0, 1.0, float(a.get("amb_buffer_pct_boost", 0.0)), 0.01, format="%.2f")
            a["amb_buffer_pct_income_boost"] = st.number_input("[Ambitieux] Bonus buffer % revenu", 0.0, 0.8, float(a.get("amb_buffer_pct_income_boost", 0.0)), 0.01, format="%.2f")

        st.session_state.assumptions = a

    with tab_results:
        st.subheader("Résultats")
        exp_m_df, exp_warn = monthlyize_expenses(st.session_state.expenses_df)
        loans_calc_df, loans_total_monthly, loan_warn = compute_loans(st.session_state.loans_df)

        errors, warns = validate_inputs(st.session_state.expenses_df, st.session_state.loans_df, st.session_state.assumptions)
        if errors:
            st.error("Erreurs:\n- " + "\n- ".join(errors))
            st.stop()

        all_warns = warns + exp_warn + loan_warn
        if all_warns:
            st.warning("Avertissements:\n- " + "\n- ".join(all_warns))

        scen_df, scen_details = compute_scenarios(exp_m_df, loans_total_monthly, st.session_state.assumptions)
        default_pick = "Confort" if "Confort" in scen_df["Scénario"].values else scen_df["Scénario"].iloc[0]
        pick = st.radio("Scénario (KPIs)", scen_df["Scénario"].tolist(), index=scen_df["Scénario"].tolist().index(default_pick), horizontal=True)

        row = scen_df[scen_df["Scénario"] == pick].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net après IR requis", f"{row['Net après IR requis']:,.0f} {CURRENCY}".replace(",", " "))
        k2.metric("Net avant IR requis", f"{row['Net avant IR requis']:,.0f} {CURRENCY}".replace(",", " "))
        k3.metric("Brut estimé requis", f"{row['Brut estimé requis']:,.0f} {CURRENCY}".replace(",", " "))
        if not math.isnan(row.get("Coût employeur estimé", np.nan)):
            k4.metric("Coût employeur estimé", f"{row['Coût employeur estimé']:,.0f} {CURRENCY}".replace(",", " "))
        else:
            k4.metric("Annuel net après IR", f"{row['Annuel (net après IR)']:,.0f} {CURRENCY}".replace(",", " "))

        st.markdown("#### Décomposition (mensuel)")
        d = scen_details[pick]["breakdown"]
        bdf = pd.DataFrame(
            [
                ("Dépenses", d["depenses"]),
                ("Prêts", d["prets"]),
                ("Épargne fixe", d["epargne_fixe"]),
                ("Buffer (montant)", d["buffer"]),
                ("Épargne % revenu", d.get("epargne_pct", 0.0)),
                ("Buffer % revenu", d.get("buffer_pct_revenu", 0.0)),
            ],
            columns=["Poste", "Montant"],
        )
        st.dataframe(bdf, use_container_width=True)

        st.markdown("#### Comparaison scénarios")
        melt = scen_df[["Scénario", "Net après IR requis", "Brut estimé requis"]].melt("Scénario", var_name="Mesure", value_name="Montant")
        chart = alt.Chart(melt).mark_bar().encode(
            x=alt.X("Scénario:N"),
            y=alt.Y("Montant:Q"),
            color="Mesure:N",
            tooltip=["Scénario", "Mesure", alt.Tooltip("Montant:Q", format=",.0f")],
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Dépenses par catégorie (scénario sélectionné)")
        active = exp_m_df[exp_m_df["actif"].fillna(True) == True].copy()
        if pick == "Minimum":
            active = active[active["niveau"].fillna("Essentiel") == "Essentiel"]
        cat = active.groupby("categorie", dropna=False)["monthly_equiv"].sum().reset_index()
        cat_chart = alt.Chart(cat).mark_bar().encode(
            x=alt.X("monthly_equiv:Q", title=f"{CURRENCY}/mois"),
            y=alt.Y("categorie:N", sort="-x", title="Catégorie"),
            tooltip=["categorie", alt.Tooltip("monthly_equiv:Q", format=",.0f")],
        )
        st.altair_chart(cat_chart, use_container_width=True)

        st.markdown("#### Table des scénarios")
        st.dataframe(scen_df, use_container_width=True)

        st.info(
            "Formule: **Net après IR = dépenses + prêts + épargne + buffer** → "
            "**Net avant IR = Net après IR / (1 - PAS)** → **Brut = Net avant IR / ratio**."
        )

    with tab_export:
        st.subheader("Exports & sauvegarde (simple)")
        exp_m_df, _ = monthlyize_expenses(st.session_state.expenses_df)
        loans_calc_df, loans_total_monthly, _ = compute_loans(st.session_state.loans_df)
        scen_df, _ = compute_scenarios(exp_m_df, loans_total_monthly, st.session_state.assumptions)

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### CSV")
            st.download_button("Dépenses (CSV)", st.session_state.expenses_df.to_csv(index=False).encode("utf-8"),
                               file_name="depenses.csv", mime="text/csv")
            st.download_button("Prêts (CSV)", st.session_state.loans_df.to_csv(index=False).encode("utf-8"),
                               file_name="prets.csv", mime="text/csv")
            st.download_button("Scénarios (CSV)", scen_df.to_csv(index=False).encode("utf-8"),
                               file_name="scenarios.csv", mime="text/csv")

        with colB:
            st.markdown("### Sauvegarde (fichier)")
            save_blob = bundle_to_json(st.session_state.expenses_df, st.session_state.loans_df, st.session_state.assumptions).encode("utf-8")
            st.download_button("Télécharger ma sauvegarde", save_blob, file_name="train_de_vie_save.json", mime="application/json")
            up = st.file_uploader("Importer une sauvegarde", type=["json"])
            if up is not None:
                try:
                    s = up.read().decode("utf-8")
                    exp, loans, a = json_to_bundle(s)
                    st.session_state.expenses_df = exp
                    st.session_state.loans_df = loans
                    st.session_state.assumptions = {**default_assumptions(), **a}
                    st.success("Sauvegarde importée.")
                except Exception as e:
                    st.error(f"Import impossible: {e}")

        st.divider()
        st.markdown("### PDF (optionnel)")
        pdf_bytes = make_pdf_summary(scen_df, st.session_state.assumptions)
        st.download_button("Télécharger résumé (PDF)", pdf_bytes, file_name="resume_salaire_requis.pdf", mime="application/pdf")

        st.divider()
        if st.button("Réinitialiser (données d'exemple)"):
            st.session_state.expenses_df = default_expenses()
            st.session_state.loans_df = default_loans()
            st.session_state.assumptions = default_assumptions()
            st.success("Réinitialisé.")


if __name__ == "__main__":
    main()
