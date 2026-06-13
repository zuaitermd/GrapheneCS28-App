import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="GrapheneCS28 Tool", page_icon="🔬", layout="wide")

# ── Encoding maps ─────────────────────────────────────────────────────────────
SYSTEM_MAP   = {"Paste": 1, "Mortar": 2, "Concrete": 3}
BINDER_MAP   = {
    "OPC": 1, "Fly Ash Geopolymer": 2, "GGBS/FA Geopolymer (1:1)": 3,
    "GGBS/FA Geopolymer Concrete": 4, "Cement/Slag Blend": 5,
    "Cement/FA Blend": 6, "Portland Pozzolanic FA Cement": 7,
    "FA/GGBS Geopolymer (FA>70%)": 8, "FA/GGBS Geopolymer Mortar": 9,
}
GRAPHENE_MAP = {"GNP": 1, "Control (no graphene)": 2, "GO": 3}
DISP_MAP     = {
    "Chemical admixture only": 1, "Hummers method": 2,
    "Ultrasonication only": 3, "Superplasticizer + ultrasonication": 4,
    "High shear mixing only": 5, "Surfactant + ultrasonication": 6,
    "Sonication + shear mixing": 7, "Mechanical + probe ultrasonication": 8,
    "PVP + high shear + probe ultrasonication": 9,
}
FEAT_NAMES = [
    "System Type", "Binder Type", "Binder Content (kg/m3)", "w/c Ratio",
    "Graphene Type", "Particle Size (um)", "Thickness (nm)",
    "Graphene Conc. (%)", "Dispersion Method"
]
CAT_SET = {"System Type", "Binder Type", "Graphene Type", "Dispersion Method"}
K_CALIB = 997.18

GA_RESULTS = {
    ("Paste","GO"):       {"Binder Type":"FA Geopolymer","Binder Content (kg/m3)":1402,"w/c":0.298,"Graphene Conc (%)":1.602,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":113.92},
    ("Paste","GNP"):      {"Binder Type":"OPC","Binder Content (kg/m3)":1408,"w/c":0.299,"Graphene Conc (%)":1.947,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":96.45},
    ("Paste","Control"):  {"Binder Type":"OPC","Binder Content (kg/m3)":1804,"w/c":0.298,"Graphene Conc (%)":0.0,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":96.87},
    ("Mortar","GO"):      {"Binder Type":"OPC","Binder Content (kg/m3)":650,"w/c":0.301,"Graphene Conc (%)":0.149,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":84.16},
    ("Mortar","GNP"):     {"Binder Type":"GGBS/FA Geo (1:1)","Binder Content (kg/m3)":695,"w/c":0.301,"Graphene Conc (%)":0.047,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":73.95},
    ("Mortar","Control"): {"Binder Type":"FA Geopolymer","Binder Content (kg/m3)":691,"w/c":0.301,"Graphene Conc (%)":0.0,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":75.94},
    ("Concrete","GO"):    {"Binder Type":"OPC","Binder Content (kg/m3)":599,"w/c":0.350,"Graphene Conc (%)":2.920,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":76.95},
    ("Concrete","GNP"):   {"Binder Type":"GGBS/FA Geo (1:1)","Binder Content (kg/m3)":600,"w/c":0.351,"Graphene Conc (%)":2.974,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":66.89},
    ("Concrete","Control"):{"Binder Type":"OPC","Binder Content (kg/m3)":593,"w/c":0.351,"Graphene Conc (%)":0.0,"Particle Size (um)":0.28,"Thickness (nm)":0.68,"Dispersion":"PVP+HS+Sonic","CS28 (MPa)":70.72},
}
SYS_MEANS = {"Paste": 43.79, "Mortar": 36.22, "Concrete": 43.75}

# ── Train models once ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models on first load (~30 sec)...")
def get_models():
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from xgboost import XGBRegressor

    df = pd.read_excel("synthetic_interpolated_SMOTE.xlsx")
    X  = df.iloc[:, :9].values.astype(float)
    y  = df.iloc[:,  9].values.astype(float)

    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(X)

    et  = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    et.fit(X, y)

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                       colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X, y)
    return imp, et, xgb

imp, et_model, xgb_model = get_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GrapheneCS28 — ML Prediction & Optimisation Tool")
st.markdown(
    "Predicting 28-day compressive strength (CS28) of graphene-reinforced cementitious composites. "
    "Trained on 127 experimental data points. "
    "Extra Trees R²=0.945 | XGBoost R²=0.897. "
    "_PhD dissertation — Khalifa University, Abu Dhabi, UAE._"
)
st.divider()

tab1, tab2, tab3 = st.tabs([
    "CS28 Prediction & SHAP",
    "Young's Modulus Estimation",
    "GA-Optimised Mix Designs",
])

# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("CS28 Prediction with SHAP Interpretability")
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("Mix Design Parameters")
        sys_t  = st.selectbox("System Type",       list(SYSTEM_MAP.keys()),   key="p_sys")
        bin_t  = st.selectbox("Binder Type",       list(BINDER_MAP.keys()),   key="p_bt")
        bin_c  = st.number_input("Binder Content (kg/m3)", 290.0, 2000.0, 425.0, 10.0, key="p_bc")
        wc     = st.number_input("w/c Ratio",               0.24,   0.60,  0.40,  0.01, key="p_wc")
        gra_t  = st.selectbox("Graphene Type",     list(GRAPHENE_MAP.keys()), key="p_gt")
        ps     = st.number_input("Particle Size (um)",       0.28,  50.0,  9.0,   0.1,  key="p_ps")
        th     = st.number_input("Thickness (nm)",           0.68, 230.0,  5.0,   0.5,  key="p_th")
        gc     = st.number_input("Graphene Conc. (% bwb)",   0.0,   4.0,  0.08,  0.01,  key="p_gc")
        dis_m  = st.selectbox("Dispersion Method", list(DISP_MAP.keys()),     key="p_dm")
        model  = st.radio("Model", ["Extra Trees", "XGBoost"], horizontal=True, key="p_mdl")
        btn    = st.button("Predict CS28", type="primary", use_container_width=True, key="p_btn")

    with col2:
        if btn:
            xr = np.array([
                SYSTEM_MAP[sys_t], BINDER_MAP[bin_t], bin_c, wc,
                GRAPHENE_MAP[gra_t], ps, th, gc, DISP_MAP[dis_m]
            ], dtype=float)
            xp = imp.transform(xr.reshape(1, -1))

            et_v  = float(et_model.predict(xp)[0])
            xgb_v = float(xgb_model.predict(xp)[0])

            c1, c2 = st.columns(2)
            c1.metric("Extra Trees", f"{et_v:.2f} MPa")
            c2.metric("XGBoost",     f"{xgb_v:.2f} MPa")

            # SHAP
            st.subheader(f"SHAP Waterfall — {model}")
            try:
                import shap
                mdl   = et_model if model == "Extra Trees" else xgb_model
                pred  = et_v     if model == "Extra Trees" else xgb_v
                exp   = shap.TreeExplainer(mdl)
                sv    = exp.shap_values(xp)[0]
                ev = exp.expected_value
                bv = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

                order  = np.argsort(np.abs(sv))
                sv_s   = sv[order]
                fn_s   = [FEAT_NAMES[i] for i in order]
                xv_s   = xr[order]
                labels = [
                    f"{f} = {int(round(v))}" if f in CAT_SET else f"{f} = {v:.3f}"
                    for f, v in zip(fn_s, xv_s)
                ]

                run = bv
                lf, rt = [], []
                for s in sv_s:
                    lf.append(min(run, run+s))
                    rt.append(max(run, run+s))
                    run += s

                fig, ax = plt.subplots(figsize=(7, 4.5))
                yp = np.arange(len(sv_s))
                for i, s in enumerate(sv_s):
                    c = "#3a86d4" if s >= 0 else "#e05c8a"
                    ax.barh(yp[i], abs(s), left=lf[i], height=0.52, color=c, edgecolor="none")
                    sign = "+" if s >= 0 else ""
                    ax.text(rt[i]+0.3 if s>=0 else lf[i]-0.3, yp[i],
                            f"{sign}{s:.2f}", va="center",
                            ha="left" if s>=0 else "right", fontsize=8, color=c)
                ax.set_yticks(yp)
                ax.set_yticklabels(labels, fontsize=8)
                ax.axvline(bv,   color="gray",  linestyle="--", linewidth=0.9, alpha=0.7)
                ax.axvline(pred, color="#333",   linestyle="--", linewidth=0.9, alpha=0.5)
                ax.text(bv,   len(sv_s)+0.2, f"E[f(x)]={bv:.1f}",  fontsize=7, color="gray")
                ax.text(pred, len(sv_s)+0.2, f"f(x)={pred:.1f}", fontsize=7, color="#333", ha="right")
                ax.set_xlabel("SHAP Value (MPa)")
                for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
                ax.grid(axis="x", alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")
        else:
            st.info("Set parameters on the left and click **Predict CS28**.")

# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("Young's Modulus (E) Estimation")
    st.markdown(
        f"**E = {K_CALIB} × √CS28** (MPa) — Physics-informed relationship "
        f"recalibrated from 42 experimental data points (Extra Trees: R²_E = 0.970, RMSE_E = 224.6 MPa)."
    )
    cs_in = st.number_input("CS28 (MPa)", 1.0, 200.0, 42.4, 0.1, key="e_cs")
    if st.button("Estimate E", type="primary", key="e_btn"):
        E_est = K_CALIB * np.sqrt(cs_in)
        st.success(f"**E = {E_est:,.0f} MPa  ({E_est/1000:.2f} GPa)**")
        cs_r = np.linspace(2, 150, 300)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(cs_r, K_CALIB*np.sqrt(cs_r)/1000, color="#1f77b4", lw=1.5,
                label=f"E = {K_CALIB}·sqrt(CS28)")
        ax.scatter([cs_in], [E_est/1000], s=80, color="#d62728", zorder=5,
                   label=f"CS28={cs_in:.1f} → E={E_est/1000:.2f} GPa")
        ax.set_xlabel("CS28 (MPa)"); ax.set_ylabel("E (GPa)")
        ax.set_title("Physics-Informed E–CS28 Relationship")
        ax.legend(fontsize=8, frameon=False)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Dataset-specific estimate. Experimental validation recommended before structural application.")

# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("GA-Optimised Mix Designs")
    st.warning("Surrogate model predictions only — not experimentally verified.")

    ca, cb = st.columns(2)
    with ca: sys_s = st.selectbox("System Type",   ["Paste","Mortar","Concrete"], key="g_sys")
    with cb: gt_s  = st.selectbox("Graphene Type", ["GO","GNP","Control"],        key="g_gt")

    if st.button("Retrieve Optimised Mix", type="primary", key="g_btn"):
        res  = GA_RESULTS[(sys_s, gt_s)]
        cs_o = res["CS28 (MPa)"]
        impr = (cs_o - SYS_MEANS[sys_s]) / SYS_MEANS[sys_s] * 100
        st.success(f"{sys_s} + {gt_s}: Predicted CS28 = {cs_o:.2f} MPa (+{impr:.1f}% above system mean)")

        rows = [(k, str(v)) for k, v in res.items() if k != "CS28 (MPa)"]
        st.dataframe(pd.DataFrame(rows, columns=["Parameter","Optimal Value"]),
                     use_container_width=True, hide_index=True)

        colors = {"GO":"#1a9641","GNP":"#2c7bb6","Control":"#d7191c"}
        gts    = ["GO","GNP","Control"]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
        for ax, sys in zip(axes, ["Paste","Mortar","Concrete"]):
            for i, gt in enumerate(gts):
                val = GA_RESULTS[(sys, gt)]["CS28 (MPa)"]
                alp = 1.0 if (sys==sys_s and gt==gt_s) else 0.4
                ax.bar(i, val, width=0.55, color=colors[gt],
                       edgecolor="black", linewidth=0.5, alpha=alp)
                ip = (val - SYS_MEANS[sys]) / SYS_MEANS[sys] * 100
                ax.text(i, val+0.8, f"+{ip:.0f}%", ha="center",
                        fontsize=7, color=colors[gt], alpha=alp)
            ax.axhline(SYS_MEANS[sys], color="black", linestyle="--", linewidth=0.8)
            ax.set_title(sys, fontweight="bold", fontsize=9)
            ax.set_xticks(range(3)); ax.set_xticklabels(gts, fontsize=8)
            ax.set_ylabel("CS28 (MPa)" if sys=="Paste" else "")
            ax.set_ylim(0, 130)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
        axes[2].legend(
            handles=[mpatches.Patch(color=colors[g], label=g) for g in gts],
            fontsize=7.5, frameon=False
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.caption(
    "GrapheneCS28 Tool | Khalifa University, Abu Dhabi | "
    "PhD dissertation 2026 | Interpolative use within training data bounds only."
)
