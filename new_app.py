import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import shap
warnings.filterwarnings('ignore')

st.set_page_config(page_title="GrapheneCS₂₈ Prediction Tool", page_icon="🔬", layout="wide")

FEATURES = ['System Type ID','Binder Type ID','Binder content (kg/m³)',
            'water-to-cement ratio','Graphene type ID',
            'Graphene particle size (micro-meter)','Graphene thickness (nano-meter)',
            'Graphene Concentration (%)','Dispersion method type ID']
TARGET = 'CS_28d_MPa'
FEAT_DISPLAY = ['System Type','Binder Type','Binder Content (kg/m³)','w/c Ratio',
                'Graphene Type','Particle Size (µm)','Thickness (nm)',
                'Graphene Conc. (%)','Dispersion Method']
CAT_FEATS = {'System Type','Binder Type','Graphene Type','Dispersion Method'}

# ── Train models (cached so only runs once per session) ───────────────────────
@st.cache_resource(show_spinner="Training models on startup — please wait ~30 seconds...")
def train_models():
    df_syn  = pd.read_excel('synthetic_interpolated_SMOTE.xlsx')
    df_real = pd.read_excel('Original_dataset_-_with_E.xlsx')
    imputer = SimpleImputer(strategy='median')
    X_syn   = imputer.fit_transform(df_syn[FEATURES].values)
    y_syn   = df_syn[TARGET].values

    et = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    et.fit(X_syn, y_syn)

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                        colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_syn, y_syn)

    et_exp  = shap.TreeExplainer(et)
    xgb_exp = shap.TreeExplainer(xgb)
    return imputer, et, xgb, et_exp, xgb_exp

imputer, et_model, xgb_model, et_explainer, xgb_explainer = train_models()

# ── Encoding maps ─────────────────────────────────────────────────────────────
SYSTEM_MAP = {'Paste': 1, 'Mortar': 2, 'Concrete': 3}
BINDER_MAP = {
    'OPC — Ordinary Portland Cement': 1, 'Fly Ash Geopolymer': 2,
    'GGBS/FA Geopolymer (1:1)': 3, 'GGBS/FA Geopolymer Concrete': 4,
    'Cement/Slag Blend (1:1)': 5, 'Cement/FA Blend': 6,
    'Portland Pozzolanic FA Cement': 7, 'FA/GGBS Geopolymer (FA>70%)': 8,
    'FA/GGBS Geopolymer Mortar': 9,
}
GRAPHENE_MAP = {'GNP — Graphene Nanoplatelets': 1, 'Control (no graphene)': 2, 'GO — Graphene Oxide': 3}
DISP_MAP = {
    'Chemical admixture only': 1, "Hummers' method": 2,
    'Ultrasonication only': 3, 'Superplasticizer + ultrasonication': 4,
    'High shear mixing only': 5, 'Surfactant + ultrasonication': 6,
    'Sonication + shear mixing': 7, 'Mechanical mixing + probe ultrasonication': 8,
    'PVP + high shear + probe ultrasonication': 9,
}
K_CALIB = 997.18
GA_RESULTS = {
    ('Paste','GO'):      {'Binder Type':'FA Geopolymer','Binder Content (kg/m³)':1402,'w/c Ratio':0.298,'Graphene Conc. (%)':1.602,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':113.92},
    ('Paste','GNP'):     {'Binder Type':'OPC','Binder Content (kg/m³)':1408,'w/c Ratio':0.299,'Graphene Conc. (%)':1.947,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':96.45},
    ('Paste','Control'): {'Binder Type':'OPC','Binder Content (kg/m³)':1804,'w/c Ratio':0.298,'Graphene Conc. (%)':0.0,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':96.87},
    ('Mortar','GO'):     {'Binder Type':'OPC','Binder Content (kg/m³)':650,'w/c Ratio':0.301,'Graphene Conc. (%)':0.149,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':84.16},
    ('Mortar','GNP'):    {'Binder Type':'GGBS/FA Geo (1:1)','Binder Content (kg/m³)':695,'w/c Ratio':0.301,'Graphene Conc. (%)':0.047,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':73.95},
    ('Mortar','Control'):{'Binder Type':'FA Geopolymer','Binder Content (kg/m³)':691,'w/c Ratio':0.301,'Graphene Conc. (%)':0.0,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':75.94},
    ('Concrete','GO'):   {'Binder Type':'OPC','Binder Content (kg/m³)':599,'w/c Ratio':0.350,'Graphene Conc. (%)':2.920,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':76.95},
    ('Concrete','GNP'):  {'Binder Type':'GGBS/FA Geo (1:1)','Binder Content (kg/m³)':600,'w/c Ratio':0.351,'Graphene Conc. (%)':2.974,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':66.89},
    ('Concrete','Control'):{'Binder Type':'OPC','Binder Content (kg/m³)':593,'w/c Ratio':0.351,'Graphene Conc. (%)':0.0,'Particle Size (µm)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS₂₈ (MPa)':70.72},
}

# ── SHAP waterfall helper ─────────────────────────────────────────────────────
def plot_shap_waterfall(shap_vals, x_raw, base_val, pred_val, title=""):
    order  = np.argsort(np.abs(shap_vals))
    sv, fn, xv = shap_vals[order], [FEAT_DISPLAY[i] for i in order], x_raw[order]
    ylabels = [f"{f} = {int(round(v))}" if f in CAT_FEATS else f"{f} = {v:.3f}"
               for f, v in zip(fn, xv)]
    running = base_val
    lefts, rights = [], []
    for s in sv:
        lefts.append(min(running, running+s))
        rights.append(max(running, running+s))
        running += s
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pos_c, neg_c = '#3a86d4', '#e05c8a'
    yp = np.arange(len(sv))
    for i, s in enumerate(sv):
        c = pos_c if s>=0 else neg_c
        ax.barh(yp[i], abs(s), left=lefts[i], height=0.52, color=c, edgecolor='none')
        sign = '+' if s>=0 else ''
        ax.text(rights[i]+0.4 if s>=0 else lefts[i]-0.4, yp[i],
                f'{sign}{s:.2f}', va='center', ha='left' if s>=0 else 'right',
                fontsize=8, color=c, fontweight='bold')
    ax.set_yticks(yp); ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.yaxis.set_tick_params(length=0, pad=4)
    ax.axvline(x=base_val, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axvline(x=pred_val, color='#333', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.text(base_val+0.3, len(sv)-0.15, f'E[f(x)]={base_val:.2f}',
            ha='left', va='bottom', fontsize=7, color='gray', style='italic')
    ax.text(pred_val-0.3, len(sv)-0.15, f'f(x)={pred_val:.2f}',
            ha='right', va='bottom', fontsize=7, color='#333', style='italic', alpha=0.7)
    ax.legend(handles=[mpatches.Patch(color=pos_c, label='Positive'),
                        mpatches.Patch(color=neg_c, label='Negative')],
              fontsize=8, frameon=True, framealpha=0.95, edgecolor='#ccc', loc='lower right')
    ax.set_xlabel('SHAP Value (MPa)', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    ax.set_xlim(min(base_val, pred_val)-10, max(base_val, pred_val)+15)
    ax.set_ylim(-0.5, len(sv)+0.8)
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linewidth=0.4)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig); plt.close()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 GrapheneCS₂₈ — Prediction & Optimisation Tool")
st.markdown("""
Predicting the **28-day compressive strength (CS₂₈)** of graphene-reinforced cementitious composites
using **Extra Trees** and **XGBoost** ML models trained on 127 experimental data points.
*PhD dissertation — Khalifa University, Abu Dhabi, UAE.*
""")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊  CS₂₈ Prediction & SHAP","🔧  Young's Modulus","🧬  GA-Optimised Mixes"])

# ─── TAB 1 ───────────────────────────────────────────────────────────────────
with tab1:
    st.header("CS₂₈ Prediction with SHAP Interpretability")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Mix Design Inputs")
        system_type   = st.selectbox("System Type",       list(SYSTEM_MAP.keys()))
        binder_type   = st.selectbox("Binder Type",       list(BINDER_MAP.keys()))
        binder_cont   = st.number_input("Binder Content (kg/m³)",  290.0, 2000.0, 425.0, 10.0)
        wc_ratio      = st.number_input("w/c Ratio",               0.24,  0.60,   0.40,  0.01)
        graphene_type = st.selectbox("Graphene Type",     list(GRAPHENE_MAP.keys()))
        particle_size = st.number_input("Particle Size (µm)",       0.28,  50.0,   9.0,   0.1)
        thickness     = st.number_input("Thickness (nm)",           0.68,  230.0,  5.0,   0.5)
        graphene_conc = st.number_input("Graphene Conc. (% bwb)",   0.0,   4.0,   0.08,  0.01)
        disp_method   = st.selectbox("Dispersion Method", list(DISP_MAP.keys()))
        model_choice  = st.radio("Explain model", ["Extra Trees","XGBoost","Both"], horizontal=True)
        predict_btn   = st.button("▶  Predict CS₂₈", type="primary", use_container_width=True)

    with c2:
        if predict_btn:
            x_raw  = np.array([SYSTEM_MAP[system_type], BINDER_MAP[binder_type],
                                binder_cont, wc_ratio, GRAPHENE_MAP[graphene_type],
                                particle_size, thickness, graphene_conc,
                                DISP_MAP[disp_method]], dtype=float)
            x_proc = imputer.transform(x_raw.reshape(1,-1))
            et_cs  = float(et_model.predict(x_proc)[0])
            xgb_cs = float(xgb_model.predict(x_proc)[0])
            m1,m2  = st.columns(2)
            m1.metric("Extra Trees (R²=0.945)", f"{et_cs:.2f} MPa")
            m2.metric("XGBoost (R²=0.897)",     f"{xgb_cs:.2f} MPa")
            if model_choice in ["Extra Trees","Both"]:
                sv = et_explainer.shap_values(x_proc)[0]
                plot_shap_waterfall(sv, x_raw, float(et_explainer.expected_value), et_cs, "SHAP Waterfall — Extra Trees")
            if model_choice in ["XGBoost","Both"]:
                sv = xgb_explainer.shap_values(x_proc)[0]
                plot_shap_waterfall(sv, x_raw, float(xgb_explainer.expected_value), xgb_cs, "SHAP Waterfall — XGBoost")
        else:
            st.info("👈 Set parameters and click **Predict CS₂₈**.")

# ─── TAB 2 ───────────────────────────────────────────────────────────────────
with tab2:
    st.header("Young's Modulus (E) Estimation")
    st.latex(r"E = 997.18\,\sqrt{f_c} \quad \text{(MPa)}")
    st.markdown("Recalibrated from 42 experimental data points. Extra Trees: R²_E = 0.970, RMSE_E = 224.6 MPa.")
    cs_in = st.number_input("CS₂₈ (MPa)", 1.0, 200.0, 42.4, 0.1)
    if st.button("▶  Estimate E", type="primary"):
        E_est = K_CALIB * np.sqrt(cs_in)
        st.success(f"**E = {E_est:,.0f} MPa  ({E_est/1000:.2f} GPa)**")
        cs_r = np.linspace(2,150,300); E_r = K_CALIB*np.sqrt(cs_r)
        fig,ax = plt.subplots(figsize=(7,3.2))
        ax.plot(cs_r, E_r/1000, color='#1f77b4', linewidth=1.5, label=f'E = {K_CALIB}·√CS₂₈')
        ax.scatter([cs_in],[E_est/1000], s=90, color='#d62728', zorder=5,
                   label=f'CS₂₈={cs_in:.1f} → E={E_est/1000:.2f} GPa')
        ax.set_xlabel('CS₂₈ (MPa)', fontsize=9); ax.set_ylabel('E (GPa)', fontsize=9)
        ax.set_title('Physics-Informed E–CS₂₈ Relationship', fontsize=9, fontweight='bold')
        ax.legend(fontsize=8,frameon=False)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.2,linewidth=0.4); fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("⚠️ Dataset-specific estimate. Experimental validation recommended before structural application.")

# ─── TAB 3 ───────────────────────────────────────────────────────────────────
with tab3:
    st.header("GA-Optimised Mix Designs")
    st.warning("⚠️ Surrogate model predictions — not experimentally verified. Experimental validation is strongly recommended.")
    ca,cb = st.columns(2)
    with ca: sys_sel = st.selectbox("System Type",  ['Paste','Mortar','Concrete'], key='ga_sys')
    with cb: gt_sel  = st.selectbox("Graphene Type",['GO','GNP','Control'],        key='ga_gt')
    if st.button("▶  Retrieve Optimised Mix", type="primary"):
        res    = GA_RESULTS[(sys_sel, gt_sel)]
        cs_opt = res['Predicted CS₂₈ (MPa)']
        means  = {'Paste':43.79,'Mortar':36.22,'Concrete':43.75}
        impr   = (cs_opt-means[sys_sel])/means[sys_sel]*100
        st.success(f"**{sys_sel} + {gt_sel}: Predicted CS₂₈ = {cs_opt:.2f} MPa (+{impr:.1f}% above system mean)**")
        rows = [(k,str(v)) for k,v in res.items() if k!='Predicted CS₂₈ (MPa)']
        st.dataframe(pd.DataFrame(rows, columns=['Parameter','Optimal Value']),
                     use_container_width=True, hide_index=True)
        systems=['Paste','Mortar','Concrete']; gts=['GO','GNP','Control']
        colors={'GO':'#1a9641','GNP':'#2c7bb6','Control':'#d7191c'}
        fig,axes = plt.subplots(1,3,figsize=(10,3.8))
        for ax,sys in zip(axes,systems):
            for i,gt in enumerate(gts):
                val=GA_RESULTS[(sys,gt)]['Predicted CS₂₈ (MPa)']
                ip=(val-means[sys])/means[sys]*100
                alp=1.0 if (sys==sys_sel and gt==gt_sel) else 0.4
                ax.bar(i,val,width=0.55,color=colors[gt],edgecolor='black',linewidth=0.5,alpha=alp)
                ax.text(i,val+0.5,f'+{ip:.0f}%',ha='center',fontsize=7,fontweight='bold',
                        color=colors[gt],alpha=alp)
            ax.axhline(means[sys],color='black',linestyle='--',linewidth=0.8)
            ax.set_title(sys,fontweight='bold',fontsize=9)
            ax.set_xticks(range(3)); ax.set_xticklabels(gts,fontsize=8)
            ax.set_ylabel('CS₂₈ (MPa)' if sys=='Paste' else '')
            ax.set_ylim(0,130)
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            ax.grid(axis='y',alpha=0.2,linewidth=0.4)
        leg=[mpatches.Patch(color=colors[g],label=g) for g in gts]
        leg.append(Line2D([0],[0],color='black',linestyle='--',linewidth=0.8,label='System mean'))
        axes[2].legend(handles=leg,fontsize=7.5,frameon=False,loc='upper left')
        fig.tight_layout(pad=1.0); st.pyplot(fig); plt.close()

st.divider()
st.caption("GrapheneCS₂₈ Tool | Khalifa University, Abu Dhabi, UAE | PhD dissertation 2026 | Interpolative use only within training data bounds.")
