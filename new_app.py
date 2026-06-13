import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GrapheneCS28 Prediction Tool",
    page_icon="🔬",
    layout="wide"
)

FEATURES = [
    'System Type ID', 'Binder Type ID', 'Binder content (kg/m3)',
    'water-to-cement ratio', 'Graphene type ID',
    'Graphene particle size (micro-meter)', 'Graphene thickness (nano-meter)',
    'Graphene Concentration (%)', 'Dispersion method type ID'
]
TARGET = 'CS_28d_MPa'

FEAT_DISPLAY = [
    'System Type', 'Binder Type', 'Binder Content (kg/m3)', 'w/c Ratio',
    'Graphene Type', 'Particle Size (um)', 'Thickness (nm)',
    'Graphene Conc. (%)', 'Dispersion Method'
]

SYSTEM_MAP   = {'Paste': 1, 'Mortar': 2, 'Concrete': 3}
BINDER_MAP   = {
    'OPC': 1, 'Fly Ash Geopolymer': 2, 'GGBS/FA Geopolymer (1:1)': 3,
    'GGBS/FA Geopolymer Concrete': 4, 'Cement/Slag Blend': 5,
    'Cement/FA Blend': 6, 'Portland Pozzolanic FA Cement': 7,
    'FA/GGBS Geopolymer (FA>70%)': 8, 'FA/GGBS Geopolymer Mortar': 9,
}
GRAPHENE_MAP = {'GNP': 1, 'Control (no graphene)': 2, 'GO': 3}
DISP_MAP     = {
    'Chemical admixture only': 1, 'Hummers method': 2,
    'Ultrasonication only': 3, 'Superplasticizer + ultrasonication': 4,
    'High shear mixing only': 5, 'Surfactant + ultrasonication': 6,
    'Sonication + shear mixing': 7, 'Mechanical + probe ultrasonication': 8,
    'PVP + high shear + probe ultrasonication': 9,
}
K_CALIB = 997.18

GA_RESULTS = {
    ('Paste','GO'):       {'Binder Type':'FA Geopolymer','Binder Content (kg/m3)':1402,'w/c':0.298,'Graphene Conc (%)':1.602,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':113.92},
    ('Paste','GNP'):      {'Binder Type':'OPC','Binder Content (kg/m3)':1408,'w/c':0.299,'Graphene Conc (%)':1.947,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':96.45},
    ('Paste','Control'):  {'Binder Type':'OPC','Binder Content (kg/m3)':1804,'w/c':0.298,'Graphene Conc (%)':0.0,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':96.87},
    ('Mortar','GO'):      {'Binder Type':'OPC','Binder Content (kg/m3)':650,'w/c':0.301,'Graphene Conc (%)':0.149,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':84.16},
    ('Mortar','GNP'):     {'Binder Type':'GGBS/FA Geo (1:1)','Binder Content (kg/m3)':695,'w/c':0.301,'Graphene Conc (%)':0.047,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':73.95},
    ('Mortar','Control'): {'Binder Type':'FA Geopolymer','Binder Content (kg/m3)':691,'w/c':0.301,'Graphene Conc (%)':0.0,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':75.94},
    ('Concrete','GO'):    {'Binder Type':'OPC','Binder Content (kg/m3)':599,'w/c':0.350,'Graphene Conc (%)':2.920,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':76.95},
    ('Concrete','GNP'):   {'Binder Type':'GGBS/FA Geo (1:1)','Binder Content (kg/m3)':600,'w/c':0.351,'Graphene Conc (%)':2.974,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':66.89},
    ('Concrete','Control'):{'Binder Type':'OPC','Binder Content (kg/m3)':593,'w/c':0.351,'Graphene Conc (%)':0.0,'Particle Size (um)':0.28,'Thickness (nm)':0.68,'Dispersion':'PVP+HS+Sonic','Predicted CS28 (MPa)':70.72},
}

@st.cache_resource(show_spinner="Training models — please wait ~60 seconds on first load...")
def train_models():
    from sklearn.ensemble import ExtraTreesRegressor
    from xgboost import XGBRegressor
    from sklearn.impute import SimpleImputer

    df_syn  = pd.read_excel('synthetic_interpolated_SMOTE.xlsx')
    df_real = pd.read_excel('Original_dataset_-_with_E.xlsx')

    # Fix column names — remove special chars for compatibility
    df_syn.columns  = [c.replace('(kg/m³)','(kg/m3)').replace('µ','').replace('µm','micro-meter') for c in df_syn.columns]
    df_real.columns = [c.replace('(kg/m³)','(kg/m3)').replace('µ','').replace('µm','micro-meter') for c in df_real.columns]

    # Find matching column names
    syn_cols  = df_syn.columns.tolist()
    real_cols = df_real.columns.tolist()

    # Use positional mapping — col order is fixed
    X_syn  = df_syn.iloc[:, :9].values.astype(float)
    y_syn  = df_syn.iloc[:, 9].values.astype(float)

    imp = SimpleImputer(strategy='median')
    X_syn = imp.fit_transform(X_syn)

    et = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    et.fit(X_syn, y_syn)

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                        colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_syn, y_syn)

    return imp, et, xgb

imputer, et_model, xgb_model = train_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GrapheneCS28 - Prediction & Optimisation Tool")
st.markdown("""
**28-day compressive strength (CS28) prediction** for graphene-reinforced cementitious composites.  
Models: Extra Trees (R2=0.945) and XGBoost (R2=0.897) | Trained on 127 experimental data points.  
*PhD dissertation — Khalifa University, Abu Dhabi, UAE.*
""")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "CS28 Prediction & SHAP",
    "Young's Modulus Estimation",
    "GA-Optimised Mix Designs"
])

# ─── TAB 1 ───────────────────────────────────────────────────────────────────
with tab1:
    st.header("CS28 Prediction")
    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.subheader("Mix Design Inputs")
        system_type   = st.selectbox("System Type",       list(SYSTEM_MAP.keys()),   key='t1_sys')
        binder_type   = st.selectbox("Binder Type",       list(BINDER_MAP.keys()),   key='t1_bt')
        binder_cont   = st.number_input("Binder Content (kg/m3)",  290.0, 2000.0, 425.0, 10.0, key='t1_bc')
        wc_ratio      = st.number_input("w/c Ratio",               0.24,  0.60,   0.40,  0.01,  key='t1_wc')
        graphene_type = st.selectbox("Graphene Type",     list(GRAPHENE_MAP.keys()), key='t1_gt')
        particle_size = st.number_input("Particle Size (um)",       0.28,  50.0,   9.0,   0.1,   key='t1_ps')
        thickness     = st.number_input("Thickness (nm)",           0.68,  230.0,  5.0,   0.5,   key='t1_th')
        graphene_conc = st.number_input("Graphene Conc. (% bwb)",   0.0,   4.0,   0.08,  0.01,  key='t1_gc')
        disp_method   = st.selectbox("Dispersion Method", list(DISP_MAP.keys()),     key='t1_dm')
        predict_btn   = st.button("Predict CS28", type="primary", use_container_width=True)

    with c2:
        if predict_btn:
            x_raw = np.array([
                SYSTEM_MAP[system_type], BINDER_MAP[binder_type],
                binder_cont, wc_ratio, GRAPHENE_MAP[graphene_type],
                particle_size, thickness, graphene_conc, DISP_MAP[disp_method]
            ], dtype=float)
            x_proc = imputer.transform(x_raw.reshape(1, -1))
            et_cs  = float(et_model.predict(x_proc)[0])
            xgb_cs = float(xgb_model.predict(x_proc)[0])

            m1, m2 = st.columns(2)
            m1.metric("Extra Trees", f"{et_cs:.2f} MPa")
            m2.metric("XGBoost",     f"{xgb_cs:.2f} MPa")

            # SHAP
            st.subheader("SHAP Feature Contributions — Extra Trees")
            try:
                import shap, matplotlib.pyplot as plt, matplotlib.patches as mp
                et_exp = shap.TreeExplainer(et_model)
                sv     = et_exp.shap_values(x_proc)[0]
                bv     = float(et_exp.expected_value)

                order = np.argsort(np.abs(sv))
                sv_s  = sv[order]
                fn_s  = [FEAT_DISPLAY[i] for i in order]
                xv_s  = x_raw[order]

                ylabels = []
                cat_set = {'System Type','Binder Type','Graphene Type','Dispersion Method'}
                for f, v in zip(fn_s, xv_s):
                    ylabels.append(f"{f} = {int(round(v))}" if f in cat_set else f"{f} = {v:.3f}")

                running = bv
                lefts, rights = [], []
                for s in sv_s:
                    lefts.append(min(running, running+s))
                    rights.append(max(running, running+s))
                    running += s

                fig, ax = plt.subplots(figsize=(7, 4.5))
                yp = np.arange(len(sv_s))
                for i, s in enumerate(sv_s):
                    c = '#3a86d4' if s >= 0 else '#e05c8a'
                    ax.barh(yp[i], abs(s), left=lefts[i], height=0.52, color=c, edgecolor='none')
                    sign = '+' if s >= 0 else ''
                    ax.text(rights[i]+0.3 if s>=0 else lefts[i]-0.3, yp[i],
                            f'{sign}{s:.2f}', va='center',
                            ha='left' if s>=0 else 'right', fontsize=8, color=c)

                ax.set_yticks(yp); ax.set_yticklabels(ylabels, fontsize=8)
                ax.axvline(bv, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
                ax.axvline(et_cs, color='black', linestyle='--', linewidth=0.9, alpha=0.5)
                ax.text(bv, len(sv_s)+0.1, f'Base={bv:.1f}', fontsize=7, color='gray')
                ax.text(et_cs, len(sv_s)+0.1, f'Pred={et_cs:.1f}', fontsize=7, color='black', ha='right')
                ax.set_xlabel('SHAP Value (MPa)')
                for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
                ax.grid(axis='x', alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")
        else:
            st.info("Set parameters on the left and click Predict CS28.")

# ─── TAB 2 ───────────────────────────────────────────────────────────────────
with tab2:
    st.header("Young's Modulus Estimation")
    st.markdown(f"**E = {K_CALIB} x sqrt(CS28)** (MPa) — Recalibrated from 42 experimental data points (R2=0.970)")
    cs_in = st.number_input("Enter CS28 (MPa)", 1.0, 200.0, 42.4, 0.1, key='t2_cs')
    if st.button("Estimate E", type="primary", key='t2_btn'):
        E_est = K_CALIB * np.sqrt(cs_in)
        st.success(f"**Estimated E = {E_est:,.0f} MPa  ({E_est/1000:.2f} GPa)**")

        import matplotlib.pyplot as plt
        cs_r = np.linspace(2, 150, 300)
        E_r  = K_CALIB * np.sqrt(cs_r)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(cs_r, E_r/1000, color='#1f77b4', linewidth=1.5)
        ax.scatter([cs_in], [E_est/1000], s=80, color='#d62728', zorder=5,
                   label=f'CS28={cs_in:.1f} MPa -> E={E_est/1000:.2f} GPa')
        ax.set_xlabel('CS28 (MPa)'); ax.set_ylabel('E (GPa)')
        ax.set_title('E = 997.18 x sqrt(CS28)')
        ax.legend(fontsize=8, frameon=False)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Disclaimer: Dataset-specific estimate for paste/mortar systems. Experimental validation recommended.")

# ─── TAB 3 ───────────────────────────────────────────────────────────────────
with tab3:
    st.header("GA-Optimised Mix Designs")
    st.warning("Surrogate model predictions — not experimentally verified.")
    ca, cb = st.columns(2)
    with ca: sys_sel = st.selectbox("System Type",  ['Paste','Mortar','Concrete'], key='t3_sys')
    with cb: gt_sel  = st.selectbox("Graphene Type",['GO','GNP','Control'],        key='t3_gt')

    if st.button("Retrieve Optimised Mix", type="primary", key='t3_btn'):
        res    = GA_RESULTS[(sys_sel, gt_sel)]
        cs_opt = res['Predicted CS28 (MPa)']
        means  = {'Paste':43.79,'Mortar':36.22,'Concrete':43.75}
        impr   = (cs_opt - means[sys_sel]) / means[sys_sel] * 100
        st.success(f"{sys_sel} + {gt_sel}: Predicted CS28 = {cs_opt:.2f} MPa (+{impr:.1f}% above system mean)")
        rows = [(k, str(v)) for k,v in res.items() if k != 'Predicted CS28 (MPa)']
        st.dataframe(pd.DataFrame(rows, columns=['Parameter','Optimal Value']),
                     use_container_width=True, hide_index=True)

        import matplotlib.pyplot as plt, matplotlib.patches as mp
        systems = ['Paste','Mortar','Concrete']
        gts     = ['GO','GNP','Control']
        colors  = {'GO':'#1a9641','GNP':'#2c7bb6','Control':'#d7191c'}
        fig, axes = plt.subplots(1,3,figsize=(9,3.5))
        for ax, sys in zip(axes, systems):
            for i, gt in enumerate(gts):
                val  = GA_RESULTS[(sys,gt)]['Predicted CS28 (MPa)']
                alp  = 1.0 if (sys==sys_sel and gt==gt_sel) else 0.4
                ax.bar(i, val, width=0.55, color=colors[gt],
                       edgecolor='black', linewidth=0.5, alpha=alp)
                ip = (val-means[sys])/means[sys]*100
                ax.text(i, val+0.5, f'+{ip:.0f}%', ha='center',
                        fontsize=7, color=colors[gt], alpha=alp)
            ax.axhline(means[sys], color='black', linestyle='--', linewidth=0.8)
            ax.set_title(sys, fontweight='bold', fontsize=9)
            ax.set_xticks(range(3)); ax.set_xticklabels(gts, fontsize=8)
            ax.set_ylabel('CS28 (MPa)' if sys=='Paste' else '')
            ax.set_ylim(0,130)
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            ax.grid(axis='y', alpha=0.2)
        leg = [mp.Patch(color=colors[g], label=g) for g in gts]
        axes[2].legend(handles=leg, fontsize=7.5, frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.caption("GrapheneCS28 Tool | Khalifa University, Abu Dhabi | PhD dissertation 2026 | For interpolative use within training data bounds only.")
