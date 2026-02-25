"""
App — Salaire requis pour soutenir un train de vie (France, salarié)

Entrées :
- Dépenses (lignes modifiables, fréquences, Essentiel/Confort)
- Crédits / dettes (multi-prêts : capital, taux, durée, assurance, frais)

Sorties :
- Budget mensuel / annuel (dépenses)
- Total mensualités crédits
- Besoin "cash" = Net après IR requis (dépenses + crédits + épargne + marge imprévus)
- Net avant IR requis (approx via PAS)
- Brut requis (approx via ratio net→brut)
- Scénarios : Minimum / Confort / Ambitieux

Philosophie :
- Calculs simples, aucune hypothèse cachée : tout est visible et modifiable.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# -----------------------
# Config
# -----------------------
APP_TITLE = "Simulateur de salaire requis — Train de vie (France)"
CURRENCY = "€"

FREQ_OPTIONS = ["mensuel", "hebdo", "trimestriel", "annuel", "ponctuel"]
LEVEL_OPTIONS = ["Essentiel", "Confort"]
LOAN_TYPE_OPTIONS = ["immo", "auto", "conso", "etudiant", "autre"]
INS_TYPE_OPTIONS = ["Aucune", "€/mois", "%/an sur capital"]


# -----------------------
# Utils
# -----------------------
def fmt_eur(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.0f} {CURRENCY}".replace(",", " ")


def as_date(x) -> Optional[date]:
    """Convertit une valeur (date/datetime/NaT/None) en date ou None."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return None


def months_between_inclusive(d1: date, d2: date) -> int:
    if d2 < d1:
        return 0
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + 1


# -----------------------
# Finance (prêt)
# -----------------------
def pmt(rate: float, nper: int, pv: float) -> float:
    """Mensualité (positive) pour un prêt pv (positif), taux périodique rate."""
    if nper <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return pv / nper
    return pv * (rate / (1 - (1 + rate) ** (-nper)))


def loan_monthly_payment(capital: float, annual_rate_pct: float, years: float) -> float:
    """Mensualité hors assurance et hors frais (échéances constantes)."""
    n = int(round(years * 12))
    r = (annual_rate_pct / 100.0) / 12.0
    return float(pmt(r, n, capital))


def amortization_schedule(capital: float, annual_rate_pct: float, years: float) -> pd.DataFrame:
    """Tableau d'amortissement (hors assurance/frais)."""
    n = int(round(years * 12))
    r = (annual_rate_pct / 100.0) / 12.0
    payment = loan_monthly_payment(capital, annual_rate_pct, years)

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
# Data schema (robust import/export)
# -----------------------
EXPENSE_COLS = [
    "nom", "categorie", "montant", "frequence", "date_debut", "date_fin",
    "niveau", "commentaire", "actif"
]

LOAN_COLS = [
    "nom", "type", "capital", "taux_annuel_pct", "duree_annees",
    "assurance_mode", "assurance_mensuelle", "assurance_taux_annuel_pct",
    "frais", "etaler_frais"
]


def ensure_expense_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=EXPENSE_COLS)
    for c in EXPENSE_COLS:
        if c not in df.columns:
            df[c] = None
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce").fillna(0.0)
    df["frequence"] = df["frequence"].fillna("mensuel")
    df["niveau"] = df["niveau"].fillna("Essentiel")
    df["actif"] = df["actif"].fillna(True).astype(bool)
    df["categorie"] = df["categorie"].fillna("Autres")
    df["nom"] = df["nom"].fillna("")
    df["commentaire"] = df["commentaire"].fillna("")
    return df[EXPENSE_COLS].copy()


def ensure_loan_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=LOAN_COLS)
    for c in LOAN_COLS:
        if c not in df.columns:
            df[c] = None
    df["capital"] = pd.to_numeric(df["capital"], errors="coerce").fillna(0.0)
    df["taux_annuel_pct"] = pd.to_numeric(df["taux_annuel_pct"], errors="coerce").fillna(0.0)
    df["duree_annees"] = pd.to_numeric(df["duree_annees"], errors="coerce").fillna(0.0)
    df["assurance_mensuelle"] = pd.to_numeric(df["assurance_mensuelle"], errors="coerce").fillna(0.0)
    df["assurance_taux_annuel_pct"] = pd.to_numeric(df["assurance_taux_annuel_pct"], errors="coerce").fillna(0.0)
    df["frais"] = pd.to_numeric(df["frais"], errors="coerce").fillna(0.0)

    df["type"] = df["type"].fillna("autre")
    df["assurance_mode"] = df["assurance_mode"].fillna("Aucune")
    df["etaler_frais"] = df["etaler_frais"].fillna(True).astype(bool)
    df["nom"] = df["nom"].fillna("")
    return df[LOAN_COLS].copy()


# -----------------------
# Monthlyize expenses
# -----------------------
def monthlyize_expenses(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = ensure_expense_schema(df).copy()

    monthly_vals = []
    annual_vals = []

    for _, row in df.iterrows():
        amount = float(row["montant"] or 0.0)
        freq = str(row["frequence"]).strip().lower()

        if freq not in FREQ_OPTIONS:
            warnings.append(f"Fréquence inconnue '{freq}' → traitée comme 'mensuel'.")
            freq = "mensuel"

        if amount < 0:
            warnings.append(f"Montant négatif sur '{row['nom']}'. (On garde, mais c’est suspect.)")

        if freq == "mensuel":
            m = amount
        elif freq == "hebdo":
            m = amount * (52 / 12)
        elif freq == "trimestriel":
            m = amount / 3
        elif freq == "annuel":
            m = amount / 12
        elif freq == "ponctuel":
            s = as_date(row["date_debut"])
            e = as_date(row["date_fin"]) or s
            if s is None:
                warnings.append(f"Ponctuel '{row['nom']}' sans date → étalé sur 12 mois.")
                months = 12
            else:
                months = months_between_inclusive(s, e) if e else 1
                if months <= 0:
                    warnings.append(f"Ponctuel '{row['nom']}' : dates incohérentes → étalé sur 12 mois.")
                    months = 12
            m = amount / months
        else:
            m = amount

        monthly_vals.append(m)
        annual_vals.append(m * 12)

    df["mensuel_equiv"] = monthly_vals
    df["annuel_equiv"] = annual_vals
    return df, warnings


# -----------------------
# Loans compute
# -----------------------
def compute_loans(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, List[str]]:
    warnings: List[str] = []
    df = ensure_loan_schema(df).copy()

    if df.empty:
        df["mensualite_hors_assurance"] = []
        df["assurance_calc"] = []
        df["frais_mensuels_calc"] = []
        df["mensualite_totale"] = []
        return df, 0.0, warnings

    pays = []
    ins = []
    fees_m = []
    totals = []

    for _, r in df.iterrows():
        name = r["nom"] or "(sans nom)"
        capital = float(r["capital"])
        rate = float(r["taux_annuel_pct"])
        years = float(r["duree_annees"])
        n = int(round(years * 12)) if years > 0 else 0

        if years <= 0 and capital > 0:
            warnings.append(f"Prêt '{name}': durée invalide (>0 requis).")
        if rate < 0:
            warnings.append(f"Prêt '{name}': taux négatif (bizarre).")

        pay = loan_monthly_payment(capital, rate, years) if (capital > 0 and years > 0) else 0.0

        mode = str(r["assurance_mode"] or "Aucune")
        if mode == "€/mois":
            insurance = float(r["assurance_mensuelle"])
        elif mode == "%/an sur capital":
            insurance = capital * (float(r["assurance_taux_annuel_pct"]) / 100.0) / 12.0
        else:
            insurance = 0.0

        fees = float(r["frais"])
        spread = bool(r["etaler_frais"])
        f_m = (fees / n) if (spread and n > 0) else 0.0

        total = pay + insurance + f_m

        pays.append(pay)
        ins.append(insurance)
        fees_m.append(f_m)
        totals.append(total)

    df["mensualite_hors_assurance"] = pays
    df["assurance_calc"] = ins
    df["frais_mensuels_calc"] = fees_m
    df["mensualite_totale"] = totals

    total_monthly = float(np.nansum(df["mensualite_totale"].values))
    return df, total_monthly, warnings


# -----------------------
# Salary logic
# -----------------------
def compute_need_net_after_ir(
    depenses_m: float,
    prets_m: float,
    epargne_fixe: float,
    epargne_pct_revenu: float,
    marge_imprevus_pct: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Besoin net après IR (cash) :
      base = depenses + prets + epargne_fixe + marge_imprevus
      si epargne_pct_revenu > 0 :
        need = base / (1 - epargne_pct_revenu)
    """
    marge = marge_imprevus_pct * depenses_m
    base = depenses_m + prets_m + epargne_fixe + marge

    if epargne_pct_revenu >= 0.95:
        return float("nan"), {}

    need = base / (1 - epargne_pct_revenu) if epargne_pct_revenu > 0 else base

    breakdown = {
        "Dépenses": depenses_m,
        "Crédits": prets_m,
        "Épargne fixe": epargne_fixe,
        "Marge imprévus": marge,
        "Épargne (% revenu)": epargne_pct_revenu * need if epargne_pct_revenu > 0 else 0.0,
    }
    return need, breakdown


def net_before_ir_from_pas(net_after_ir: float, pas_rate: float) -> float:
    return net_after_ir / (1 - pas_rate) if pas_rate < 1 else float("nan")


def brut_from_ratio(net_before_ir: float, ratio_net_brut: float) -> float:
    return net_before_ir / ratio_net_brut if ratio_net_brut > 0 else float("nan")


# -----------------------
# Defaults
# -----------------------
def default_expenses() -> pd.DataFrame:
    data = [
        # Logement
        {"nom": "Loyer", "categorie": "Logement", "montant": 1500.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Charges", "categorie": "Logement", "montant": 150.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Énergie (élec/gaz)", "categorie": "Logement", "montant": 120.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Internet", "categorie": "Logement", "montant": 40.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Assurance habitation", "categorie": "Assurances", "montant": 18.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        # Vie courante
        {"nom": "Courses", "categorie": "Alimentation", "montant": 450.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Restaurants", "categorie": "Alimentation", "montant": 220.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Confort", "commentaire": "", "actif": True},
        {"nom": "Transport (Navigo)", "categorie": "Transport", "montant": 86.4, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Essentiel", "commentaire": "", "actif": True},
        {"nom": "Sport", "categorie": "Abonnements", "montant": 45.0, "frequence": "mensuel", "date_debut": None, "date_fin": None, "niveau": "Confort", "commentaire": "", "actif": True},
        {"nom": "Vacances (mensualisées)", "categorie": "Loisirs", "montant": 1800.0, "frequence": "annuel", "date_debut": None, "date_fin": None, "niveau": "Confort", "commentaire": "", "actif": True},
    ]
    return ensure_expense_schema(pd.DataFrame(data))


def default_loans() -> pd.DataFrame:
    data = [
        {
            "nom": "Prêt auto (exemple)",
            "type": "auto",
            "capital": 15000.0,
            "taux_annuel_pct": 4.2,
            "duree_annees": 5.0,
            "assurance_mode": "€/mois",
            "assurance_mensuelle": 20.0,
            "assurance_taux_annuel_pct": 0.0,
            "frais": 300.0,
            "etaler_frais": True,
        }
    ]
    return ensure_loan_schema(pd.DataFrame(data))


def default_assumptions() -> Dict:
    return {
        # tes choix
        "pas_rate": 0.081,          # 8,1%
        "ratio_net_brut": 0.78,     # approximation (modifiable)
        "epargne_fixe": 300.0,      # €/mois
        "epargne_pct_revenu": 0.0,  # option avancée
        "marge_imprevus_pct": 0.10, # 10%
        # ambitieux
        "amb_bonus_epargne": 200.0, # +200 €/mois
        "amb_bonus_marge": 0.00,    # +0% (simple)
    }


# -----------------------
# Save / Load (simple)
# -----------------------
def bundle_to_json(expenses_df: pd.DataFrame, loans_df: pd.DataFrame, assumptions: Dict) -> str:
    payload = {
        "version": 2,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "assumptions": assumptions,
        "expenses": ensure_expense_schema(expenses_df).to_dict(orient="records"),
        "loans": ensure_loan_schema(loans_df).to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def json_to_bundle(s: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    obj = json.loads(s)
    exp = ensure_expense_schema(pd.DataFrame(obj.get("expenses", [])))
    loans = ensure_loan_schema(pd.DataFrame(obj.get("loans", [])))
    assumptions = obj.get("assumptions", {})
    return exp, loans, assumptions


# -----------------------
# UI
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
    st.caption(
        "On part du **net après IR (cash sur le compte)**, puis on remonte vers **net avant IR** et **brut**. "
        "Modèle volontairement simple : PAS et ratio net→brut sont des approximations ajustables."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["1) Mes dépenses", "2) Mes crédits", "3) Hypothèses", "4) Résultats", "5) Export / reprise"]
    )

    # --- Dépenses ---
    with tab1:
        st.subheader("Mes dépenses")
        st.write("Ajoute/supprime des lignes. La colonne **Essentiel/Confort** pilote les scénarios.")

        exp = ensure_expense_schema(st.session_state.expenses_df)
        exp_editor = st.data_editor(
            exp,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nom": st.column_config.TextColumn("Nom", required=True),
                "categorie": st.column_config.TextColumn("Catégorie", required=True),
                "montant": st.column_config.NumberColumn(f"Montant ({CURRENCY})", min_value=0.0, step=10.0, format="%.2f"),
                "frequence": st.column_config.SelectboxColumn("Fréquence", options=FREQ_OPTIONS, required=True),
                "date_debut": st.column_config.DateColumn("Début (si ponctuel)", format="DD/MM/YYYY"),
                "date_fin": st.column_config.DateColumn("Fin (si ponctuel)", format="DD/MM/YYYY"),
                "niveau": st.column_config.SelectboxColumn("Essentiel/Confort", options=LEVEL_OPTIONS, required=True),
                "commentaire": st.column_config.TextColumn("Note (optionnel)"),
                "actif": st.column_config.CheckboxColumn("Actif"),
            },
        )
        st.session_state.expenses_df = ensure_expense_schema(exp_editor)

        exp_m, exp_warn = monthlyize_expenses(st.session_state.expenses_df)
        st.markdown("#### Équivalents calculés")
        st.dataframe(
            exp_m[["nom", "categorie", "niveau", "frequence", "montant", "mensuel_equiv", "annuel_equiv", "actif"]],
            use_container_width=True,
        )
        if exp_warn:
            st.warning("Avertissements :\n- " + "\n- ".join(exp_warn))

    # --- Crédits ---
    with tab2:
        st.subheader("Mes crédits / dettes")
        st.write("Tu peux avoir plusieurs prêts : immo, auto, conso, etc.")

        loans = ensure_loan_schema(st.session_state.loans_df)
        loans_editor = st.data_editor(
            loans,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nom": st.column_config.TextColumn("Nom", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=LOAN_TYPE_OPTIONS, required=True),
                "capital": st.column_config.NumberColumn(f"Capital ({CURRENCY})", min_value=0.0, step=100.0, format="%.2f"),
                "taux_annuel_pct": st.column_config.NumberColumn("Taux annuel (%)", min_value=0.0, step=0.1, format="%.2f"),
                "duree_annees": st.column_config.NumberColumn("Durée (années)", min_value=0.1, step=0.5, format="%.1f"),
                "assurance_mode": st.column_config.SelectboxColumn("Assurance", options=INS_TYPE_OPTIONS, required=True),
                "assurance_mensuelle": st.column_config.NumberColumn(f"Assurance ({CURRENCY}/mois)", min_value=0.0, step=1.0, format="%.2f"),
                "assurance_taux_annuel_pct": st.column_config.NumberColumn("Assurance (%/an)", min_value=0.0, step=0.05, format="%.2f"),
                "frais": st.column_config.NumberColumn(f"Frais uniques ({CURRENCY})", min_value=0.0, step=10.0, format="%.2f"),
                "etaler_frais": st.column_config.CheckboxColumn("Étaler les frais"),
            },
        )
        st.session_state.loans_df = ensure_loan_schema(loans_editor)

        loans_calc, loans_total, loan_warn = compute_loans(st.session_state.loans_df)

        st.markdown("#### Mensualités calculées")
        if loans_calc.empty:
            st.info("Aucun crédit renseigné.")
        else:
            st.dataframe(
                loans_calc[
                    ["nom", "type", "capital", "taux_annuel_pct", "duree_annees",
                     "mensualite_hors_assurance", "assurance_calc", "frais_mensuels_calc", "mensualite_totale"]
                ],
                use_container_width=True,
            )
            st.metric("Total mensualités crédits", fmt_eur(loans_total))

        if loan_warn:
            st.warning("Avertissements :\n- " + "\n- ".join(loan_warn))

        with st.expander("Option : tableau d’amortissement (pour un prêt)"):
            if not loans_calc.empty:
                pick = st.selectbox("Choisir un prêt", loans_calc["nom"].fillna("(sans nom)").tolist())
                row = loans_calc[loans_calc["nom"] == pick].iloc[0]
                st.dataframe(
                    amortization_schedule(float(row["capital"]), float(row["taux_annuel_pct"]), float(row["duree_annees"])),
                    use_container_width=True,
                )

    # --- Hypothèses ---
    with tab3:
        st.subheader("Hypothèses (simples)")

        a = st.session_state.assumptions

        c1, c2, c3 = st.columns(3)
        with c1:
            a["pas_rate"] = st.number_input("Taux PAS (ex : 0.081 = 8,1%)", 0.0, 0.99, float(a.get("pas_rate", 0.081)), 0.001, format="%.3f")
        with c2:
            a["ratio_net_brut"] = st.number_input("Ratio net avant IR → brut (approx)", 0.30, 0.95, float(a.get("ratio_net_brut", 0.78)), 0.01, format="%.2f")
        with c3:
            a["marge_imprevus_pct"] = st.number_input("Marge imprévus (ex : 0.10 = 10% des dépenses)", 0.0, 1.0, float(a.get("marge_imprevus_pct", 0.10)), 0.01, format="%.2f")

        st.divider()

        d1, d2 = st.columns(2)
        with d1:
            a["epargne_fixe"] = st.number_input(f"Épargne cible ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("epargne_fixe", 300.0)), 50.0, format="%.0f")
        with d2:
            with st.expander("Avancé (optionnel)"):
                a["epargne_pct_revenu"] = st.number_input("Épargne en % du net après IR (0.10 = 10%)", 0.0, 0.8, float(a.get("epargne_pct_revenu", 0.0)), 0.01, format="%.2f")
                st.caption("Si tu mets un %, le besoin est ajusté automatiquement (équation simple).")

        st.divider()
        st.markdown("### Scénario Ambitieux")
        a["amb_bonus_epargne"] = st.number_input(f"Ambitieux : + épargne ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("amb_bonus_epargne", 200.0)), 50.0, format="%.0f")
        a["amb_bonus_marge"] = st.number_input("Ambitieux : + marge imprévus (ex : 0.05 = +5%)", 0.0, 1.0, float(a.get("amb_bonus_marge", 0.0)), 0.01, format="%.2f")

        st.session_state.assumptions = a

    # --- Résultats ---
    with tab4:
        st.subheader("Résultats (mensuel + annuel)")

        exp_m, exp_warn = monthlyize_expenses(st.session_state.expenses_df)
        loans_calc, loans_total, loan_warn = compute_loans(st.session_state.loans_df)
        a = st.session_state.assumptions

        # validations minimales
        errors = []
        if not (0 <= float(a["pas_rate"]) < 1):
            errors.append("PAS invalide (doit être entre 0 et 0.99).")
        if not (0 < float(a["ratio_net_brut"]) <= 1):
            errors.append("Ratio net→brut invalide (doit être >0 et ≤1).")

        if errors:
            st.error("Erreurs :\n- " + "\n- ".join(errors))
            st.stop()

        if exp_warn or loan_warn:
            st.warning("Avertissements :\n- " + "\n- ".join(exp_warn + loan_warn))

        active = exp_m[exp_m["actif"] == True].copy()
        depenses_total_m = float(active["mensuel_equiv"].sum())

        essentials = active[active["niveau"] == "Essentiel"]
        depenses_ess_m = float(essentials["mensuel_equiv"].sum())

        # scénarios
        scenarios = []

        def build_scenario(name: str, depenses_m: float, epargne_fix: float, marge_pct: float):
            need_after, breakdown = compute_need_net_after_ir(
                depenses_m=depenses_m,
                prets_m=loans_total,
                epargne_fixe=epargne_fix,
                epargne_pct_revenu=float(a.get("epargne_pct_revenu", 0.0)),
                marge_imprevus_pct=marge_pct,
            )
            net_before = net_before_ir_from_pas(need_after, float(a["pas_rate"]))
            brut = brut_from_ratio(net_before, float(a["ratio_net_brut"]))
            return {
                "Scénario": name,
                "Dépenses (mensualisées)": depenses_m,
                "Crédits": loans_total,
                "Épargne": epargne_fix,
                "Marge imprévus": marge_pct * depenses_m,
                "Net après IR requis": need_after,
                "Net avant IR requis": net_before,
                "Brut requis (approx)": brut,
                "Annuel net après IR": need_after * 12 if not math.isnan(need_after) else np.nan,
                "Annuel brut": brut * 12 if not math.isnan(brut) else np.nan,
                "_breakdown": breakdown,
            }

        scenarios.append(build_scenario("Minimum", depenses_ess_m, float(a["epargne_fixe"]), float(a["marge_imprevus_pct"])))
        scenarios.append(build_scenario("Confort", depenses_total_m, float(a["epargne_fixe"]), float(a["marge_imprevus_pct"])))
        scenarios.append(
            build_scenario(
                "Ambitieux",
                depenses_total_m,
                float(a["epargne_fixe"]) + float(a.get("amb_bonus_epargne", 0.0)),
                float(a["marge_imprevus_pct"]) + float(a.get("amb_bonus_marge", 0.0)),
            )
        )

        scen_df = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in scenarios])

        pick = st.radio("Choisir le scénario", scen_df["Scénario"].tolist(), horizontal=True, index=1)
        s = next(x for x in scenarios if x["Scénario"] == pick)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net après IR requis", fmt_eur(s["Net après IR requis"]))
        k2.metric("Net avant IR requis", fmt_eur(s["Net avant IR requis"]))
        k3.metric("Brut requis (approx)", fmt_eur(s["Brut requis (approx)"]))
        k4.metric("Annuel net après IR", fmt_eur(s["Annuel net après IR"]))

        st.markdown("#### Décomposition (mensuel)")
        bdf = pd.DataFrame(list(s["_breakdown"].items()), columns=["Poste", "Montant"])
        st.dataframe(bdf, use_container_width=True)

        st.markdown("#### Comparaison des scénarios")
        melt = scen_df[["Scénario", "Net après IR requis", "Brut requis (approx)"]].melt(
            "Scénario", var_name="Mesure", value_name="Montant"
        )
        chart = alt.Chart(melt).mark_bar().encode(
            x=alt.X("Scénario:N"),
            y=alt.Y("Montant:Q"),
            color="Mesure:N",
            tooltip=["Scénario", "Mesure", alt.Tooltip("Montant:Q", format=",.0f")],
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Dépenses par catégorie")
        view = essentials if pick == "Minimum" else active
        cat = view.groupby("categorie", dropna=False)["mensuel_equiv"].sum().reset_index()
        cat_chart = alt.Chart(cat).mark_bar().encode(
            x=alt.X("mensuel_equiv:Q", title=f"{CURRENCY}/mois"),
            y=alt.Y("categorie:N", sort="-x", title="Catégorie"),
            tooltip=["categorie", alt.Tooltip("mensuel_equiv:Q", format=",.0f")],
        )
        st.altair_chart(cat_chart, use_container_width=True)

        st.markdown("#### Tableau complet")
        st.dataframe(scen_df, use_container_width=True)

        st.info(
            "Formules : "
            "**Net après IR = dépenses + crédits + épargne + marge imprévus**. "
            "Puis **Net avant IR = Net après IR / (1 - PAS)**, et **Brut ≈ Net avant IR / ratio net→brut**."
        )

    # --- Export / reprise ---
    with tab5:
        st.subheader("Export / reprise (simple)")

        exp = ensure_expense_schema(st.session_state.expenses_df)
        loans = ensure_loan_schema(st.session_state.loans_df)
        exp_m, _ = monthlyize_expenses(exp)
        loans_calc, loans_total, _ = compute_loans(loans)

        # recalcul scénarios pour export
        active = exp_m[exp_m["actif"] == True].copy()
        essentials = active[active["niveau"] == "Essentiel"]
        dep_total_m = float(active["mensuel_equiv"].sum())
        dep_ess_m = float(essentials["mensuel_equiv"].sum())

        a = st.session_state.assumptions

        def scenario_row(dep_m, epargne_fix, marge_pct, name):
            need_after, _ = compute_need_net_after_ir(dep_m, loans_total, epargne_fix, float(a.get("epargne_pct_revenu", 0.0)), marge_pct)
            net_before = net_before_ir_from_pas(need_after, float(a["pas_rate"]))
            brut = brut_from_ratio(net_before, float(a["ratio_net_brut"]))
            return {
                "Scénario": name,
                "Net après IR requis": need_after,
                "Net avant IR requis": net_before,
                "Brut requis (approx)": brut,
            }

        scen_df = pd.DataFrame([
            scenario_row(dep_ess_m, float(a["epargne_fixe"]), float(a["marge_imprevus_pct"]), "Minimum"),
            scenario_row(dep_total_m, float(a["epargne_fixe"]), float(a["marge_imprevus_pct"]), "Confort"),
            scenario_row(dep_total_m, float(a["epargne_fixe"]) + float(a.get("amb_bonus_epargne", 0.0)),
                         float(a["marge_imprevus_pct"]) + float(a.get("amb_bonus_marge", 0.0)), "Ambitieux"),
        ])

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### CSV")
            st.download_button("Télécharger dépenses (CSV)", exp.to_csv(index=False).encode("utf-8"), "depenses.csv", "text/csv")
            st.download_button("Télécharger crédits (CSV)", loans.to_csv(index=False).encode("utf-8"), "credits.csv", "text/csv")
            st.download_button("Télécharger scénarios (CSV)", scen_df.to_csv(index=False).encode("utf-8"), "scenarios.csv", "text/csv")

        with colB:
            st.markdown("### Fichier de reprise")
            payload = bundle_to_json(exp, loans, a).encode("utf-8")
            st.download_button("Télécharger mon fichier de reprise", payload, "reprise_train_de_vie.json", "application/json")

            up = st.file_uploader("Importer un fichier de reprise", type=["json"])
            if up is not None:
                try:
                    s = up.read().decode("utf-8")
                    exp2, loans2, a2 = json_to_bundle(s)
                    st.session_state.expenses_df = exp2
                    st.session_state.loans_df = loans2
                    st.session_state.assumptions = {**default_assumptions(), **a2}
                    st.success("Reprise importée.")
                except Exception as e:
                    st.error(f"Import impossible : {e}")

        st.divider()
        if st.button("Réinitialiser (données d'exemple)"):
            st.session_state.expenses_df = default_expenses()
            st.session_state.loans_df = default_loans()
            st.session_state.assumptions = default_assumptions()
            st.success("Réinitialisé.")


if __name__ == "__main__":
    main()
