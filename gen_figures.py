import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# 이 리포는 자기완결적 스냅샷(results_ansatz_uttn/results_final.json, N=5개/25 records)을 쓴다.
# Table I·Fig 1과 수치가 반드시 일치해야 하므로 다른 폴더의 결과를 섞어 읽지 않도록 주의.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))    # repo root
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_ansatz_uttn')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

SU4_DIR = os.path.join(SCRIPT_DIR, 'results_ansatz_usu4')

# 결과 로드
path = os.path.join(RESULTS_DIR, 'results_final.json')
with open(path) as f:
    data = json.load(f)

su4_path = os.path.join(SU4_DIR, 'results_final.json')
with open(su4_path) as f:
    su4_data = json.load(f)

N_values = sorted(set(r['N'] for r in data))
print(f'Loaded {len(data)} records | N={N_values}')

# --- 5-앤자츠 전체 스윕(NSL-KDD) + 3데이터셋 + 대N 확장 결과 로드 (2026-08 신규) ---
def load_json(path):
    with open(path) as f:
        return json.load(f)

ANSATZ_PARAMS = {'U_TTN': 12, 'U_15': 18, 'U_SO4': 24, 'U_5': 36, 'U_SU4': 51}
ANSATZ_STYLE = {
    'U_TTN': dict(color='#1f77b4', marker='o'),
    'U_15':  dict(color='#9467bd', marker='v'),
    'U_SO4': dict(color='#8c564b', marker='P'),
    'U_5':   dict(color='#17becf', marker='X'),
    'U_SU4': dict(color='#ff7f0e', marker='D'),
}

nslkdd_ansatz_data = {
    'U_TTN': data,
    'U_15':  load_json(os.path.join(SCRIPT_DIR, 'results_ansatz_u15', 'results_final.json')),
    'U_SO4': load_json(os.path.join(SCRIPT_DIR, 'results_ansatz_uso4', 'results_final.json')),
    'U_5':   load_json(os.path.join(SCRIPT_DIR, 'results_ansatz_u5', 'results_final.json')),
    'U_SU4': su4_data,
}

toniot_ansatz_data = {
    'U_TTN': load_json(os.path.join(SCRIPT_DIR, 'results_toniot_uttn', 'results_final.json')),
    'U_15':  load_json(os.path.join(SCRIPT_DIR, 'results_toniot_u15', 'results_final.json')),
    'U_SO4': load_json(os.path.join(SCRIPT_DIR, 'results_toniot_uso4', 'results_final.json')),
    'U_5':   load_json(os.path.join(SCRIPT_DIR, 'results_toniot_u5', 'results_final.json')),
    'U_SU4': load_json(os.path.join(SCRIPT_DIR, 'results_toniot_usu4', 'results_final.json')),
}

cicids_ansatz_data = {
    'U_TTN': load_json(os.path.join(SCRIPT_DIR, 'results_cicids_uttn', 'results_final.json')),
    'U_15':  load_json(os.path.join(SCRIPT_DIR, 'results_cicids_u15', 'results_final.json')),
    'U_SO4': load_json(os.path.join(SCRIPT_DIR, 'results_cicids_uso4', 'results_final.json')),
    'U_5':   load_json(os.path.join(SCRIPT_DIR, 'results_cicids_u5', 'results_final.json')),
    'U_SU4': load_json(os.path.join(SCRIPT_DIR, 'results_cicids_usu4', 'results_final.json')),
}

largeN_uttn = load_json(os.path.join(SCRIPT_DIR, 'results_uttn_largeN', 'results_final.json'))
largeN_u5   = load_json(os.path.join(SCRIPT_DIR, 'results_u5_largeN', 'results_final.json'))

models = ['QCNN', 'CNN', 'SVM']

def agg(data, model, metric):
    result = {}
    for N in N_values:
        vals = [r[model][metric] for r in data if r['N'] == N and model in r]
        if vals:
            result[N] = (np.mean(vals), np.std(vals))
    return result

# 스타일
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

MODEL_STYLE = {
    'QCNN': dict(color='#1f77b4', marker='o', linestyle='-',  label='QCNN'),
    'CNN':  dict(color='#d62728', marker='s', linestyle='--', label='CNN-1D'),
    'SVM':  dict(color='#2ca02c', marker='^', linestyle=':',  label='SVM (RBF)'),
}
# Fig.1/3에서 동그라미(QCNN)·네모(CNN) 마커만 축소 (세모는 기본 크기 유지)
MARKERSIZE = {'QCNN': 4.5, 'CNN': 4.5, 'SVM': 6}


def errorbar_dashed(ax, x, y, yerr, color, marker, linestyle, label, markersize=6):
    """편차를 점선 세로 막대로 표시하는 errorbar. 평균을 잇는 선(linestyle)은 그대로 두고,
    세로 오차막대(barlinecols)만 점선으로 바꾼다 — '점선 = 편차'라는 요청 반영."""
    plotline, caplines, barlinecols = ax.errorbar(
        x, y, yerr=yerr, color=color, marker=marker, linestyle=linestyle,
        label=label, markersize=markersize, capsize=3, capthick=1.2, elinewidth=1.2,
    )
    for col in barlinecols:
        col.set_linestyle((0, (4, 2)))  # 점선(dash) 패턴
    return plotline, caplines, barlinecols


# ============================================================
# Figure 1 — F1 vs N
# ============================================================

# --- (A) 원본: 음영(mean±std band) 버전. 그대로 보존. ---
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for m, sty in MODEL_STYLE.items():
    f1_agg = agg(data, m, 'f1')
    Ns = np.array(sorted(f1_agg.keys()))
    mu = np.array([f1_agg[n][0] for n in Ns])
    sd = np.array([f1_agg[n][1] for n in Ns])
    ax.plot(Ns, mu, **sty)
    ax.fill_between(Ns, mu - sd, mu + sd, alpha=0.15, color=sty['color'])
ax.set_xscale('log')
ax.set_xticks(N_values)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Training Set Size $N$')
ax.set_ylabel('F1 Score')
ax.set_title('Data Efficiency on NSL-KDD (Binary Classification)')
ax.legend(loc='lower right')
ax.set_ylim(0, 1.02)
ax.grid(True, which='major', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_f1.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_f1.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig1_f1.pdf / .png (음영 버전, 원본 유지)')

# --- (B) 대안: 점선 오차막대 버전. 지터 없이 정확한 N에 표시. 교수님 확인용 별도 파일. ---
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for m, sty in MODEL_STYLE.items():
    f1_agg = agg(data, m, 'f1')
    Ns = np.array(sorted(f1_agg.keys()))
    mu = np.array([f1_agg[n][0] for n in Ns])
    sd = np.array([f1_agg[n][1] for n in Ns])
    errorbar_dashed(ax, Ns, mu, sd, sty['color'], sty['marker'], sty['linestyle'],
                     sty['label'], markersize=MARKERSIZE[m])
ax.set_xscale('log')
ax.set_xticks(N_values)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Training Set Size $N$')
ax.set_ylabel('F1 Score')
ax.set_title('Data Efficiency on NSL-KDD (Binary Classification)')
ax.legend(loc='lower right')
ax.set_ylim(0, 1.02)
ax.grid(True, which='major', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_f1_errorbar.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_f1_errorbar.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig1_f1_errorbar.pdf / .png (점선 오차막대 버전, 신규 — 컨펌용)')

# ============================================================
# Figure 2 — Generalization Gap vs N
# ============================================================
def agg_gap(records):
    result = {}
    for N in N_values:
        vals = [r['QCNN'].get('gen_gap_f1') for r in records
                if r['N'] == N and 'QCNN' in r and r['QCNN'].get('gen_gap_f1') is not None]
        if vals:
            result[N] = (np.mean(vals), np.std(vals))
    return result

gap_ttn = agg_gap(data)
gap_su4 = agg_gap(su4_data)
Ns = np.array(sorted(gap_ttn.keys()))
mu_t = np.array([gap_ttn[n][0] for n in Ns]); sd_t = np.array([gap_ttn[n][1] for n in Ns])
mu_s = np.array([gap_su4[n][0] for n in Ns]); sd_s = np.array([gap_su4[n][1] for n in Ns])

# --- (A) 원본: 음영 버전 ---
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(Ns, mu_t, color='#1f77b4', marker='o', linestyle='-', label='U_TTN (12 params)')
ax.fill_between(Ns, mu_t - sd_t, mu_t + sd_t, alpha=0.15, color='#1f77b4')
ax.plot(Ns, mu_s, color='#ff7f0e', marker='D', linestyle='-', label='U_SU4 (51 params)')
ax.fill_between(Ns, mu_s - sd_s, mu_s + sd_s, alpha=0.15, color='#ff7f0e')
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xscale('log')
ax.set_xticks(N_values)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Training Set Size $N$')
ax.set_ylabel('Generalization Gap (Train$-$Test F1)')
ax.set_title('QCNN Generalization Gap vs. Training Size')
ax.legend(loc='upper right')
ax.grid(True, which='major', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_gen_gap.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_gen_gap.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig2_gen_gap.pdf / .png (음영 버전, U_TTN+U_SU4 갱신)')

# --- (B) 대안: 점선 오차막대 버전 ---
fig, ax = plt.subplots(figsize=(4.5, 3.2))
errorbar_dashed(ax, Ns, mu_t, sd_t, '#1f77b4', 'o', '-', 'U_TTN (12 params)', markersize=4.5)
errorbar_dashed(ax, Ns, mu_s, sd_s, '#ff7f0e', 'D', '-', 'U_SU4 (51 params)', markersize=4.5)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xscale('log')
ax.set_xticks(N_values)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Training Set Size $N$')
ax.set_ylabel('Generalization Gap (Train$-$Test F1)')
ax.set_title('QCNN Generalization Gap vs. Training Size')
ax.legend(loc='upper right')
ax.grid(True, which='major', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_gen_gap_errorbar.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_gen_gap_errorbar.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig2_gen_gap_errorbar.pdf / .png (점선 오차막대 버전, U_TTN+U_SU4 갱신)')

# ============================================================
# Figure 3 — Ansatz comparison: full five-ansatz sweep (NSL-KDD)
# ============================================================
Ns = np.array(N_values)
cnn_f1 = agg(data, 'CNN', 'f1')
svm_f1 = agg(data, 'SVM', 'f1')
mu_cnn = np.array([cnn_f1[n][0] for n in Ns])
mu_svm = np.array([svm_f1[n][0] for n in Ns])

ansatz_f1 = {a: agg(d, 'QCNN', 'f1') for a, d in nslkdd_ansatz_data.items()}
ansatz_mu = {a: np.array([ansatz_f1[a][n][0] for n in Ns]) for a in ANSATZ_STYLE}
ansatz_sd = {a: np.array([ansatz_f1[a][n][1] for n in Ns]) for a in ANSATZ_STYLE}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
ax1.plot(Ns, mu_cnn, color='#d62728', marker='s', linestyle='--', linewidth=1.1,
         markersize=4, alpha=0.55, label='CNN-1D (reference)')
ax1.plot(Ns, mu_svm, color='#2ca02c', marker='^', linestyle=':', linewidth=1.1,
         markersize=4, alpha=0.55, label='SVM (reference)')
for a, sty in ANSATZ_STYLE.items():
    ax1.plot(Ns, ansatz_mu[a], color=sty['color'], marker=sty['marker'], linestyle='-',
              markersize=5, label=f'{a} ({ANSATZ_PARAMS[a]}p)')
ax1.set_xscale('log'); ax1.set_xticks(N_values)
ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax1.set_xlabel('Training Set Size $N$'); ax1.set_ylabel('Test F1')
ax1.set_title('(a) Mean F1, All Five Ansätze (NSL-KDD)')
ax1.legend(loc='lower right', fontsize=6, ncol=2)
ax1.set_ylim(0, 1.0); ax1.grid(True, which='major', linestyle='--', alpha=0.4)

for a, sty in ANSATZ_STYLE.items():
    ax2.plot(Ns, ansatz_sd[a], color=sty['color'], marker=sty['marker'], linestyle='-',
              markersize=5, label=f'{a} ({ANSATZ_PARAMS[a]}p)')
ax2.set_xscale('log'); ax2.set_xticks(N_values)
ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.set_xlabel('Training Set Size $N$'); ax2.set_ylabel('F1 Std. Dev. (run-to-run)')
ax2.set_title('(b) Run-to-Run Variance'); ax2.legend(loc='upper right', fontsize=6, ncol=2)
ax2.set_ylim(0, None); ax2.grid(True, which='major', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_ansatz_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_ansatz_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig3_ansatz_comparison.pdf / .png (5-앤자츠 전체 스윕으로 확장, 2026-08)')

# ============================================================
# Figure 4 — Cross-dataset comparison (NSL-KDD, ToN_IoT, CICIDS-2017)
# ============================================================
DATASET_ANSATZ_DATA = {
    'NSL-KDD': nslkdd_ansatz_data,
    'ToN_IoT': toniot_ansatz_data,
    'CICIDS-2017': cicids_ansatz_data,
}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, (ds_name, ansatz_data) in zip(axes, DATASET_ANSATZ_DATA.items()):
    cnn_agg = agg(ansatz_data['U_TTN'], 'CNN', 'f1')
    svm_agg = agg(ansatz_data['U_TTN'], 'SVM', 'f1')
    mu_cnn_ds = np.array([cnn_agg[n][0] for n in Ns])
    mu_svm_ds = np.array([svm_agg[n][0] for n in Ns])
    # 앤자츠 5개 평균 QCNN F1 + 앤자츠 간 범위(최소~최대)를 음영으로 표시
    qcnn_by_ansatz = np.array([[agg(ansatz_data[a], 'QCNN', 'f1')[n][0] for n in Ns] for a in ANSATZ_STYLE])
    qcnn_mean = qcnn_by_ansatz.mean(axis=0)
    qcnn_min = qcnn_by_ansatz.min(axis=0)
    qcnn_max = qcnn_by_ansatz.max(axis=0)

    ax.plot(Ns, mu_cnn_ds, color='#d62728', marker='s', linestyle='--', markersize=4, label='CNN-1D')
    ax.plot(Ns, mu_svm_ds, color='#2ca02c', marker='^', linestyle=':', markersize=4, label='SVM')
    ax.plot(Ns, qcnn_mean, color='#1f77b4', marker='o', linestyle='-', markersize=4.5, label='QCNN (5-ansatz mean)')
    ax.fill_between(Ns, qcnn_min, qcnn_max, alpha=0.15, color='#1f77b4', label='QCNN (ansatz range)')
    ax.set_xscale('log'); ax.set_xticks(N_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Training Set Size $N$')
    ax.set_title(ds_name)
    ax.set_ylim(0, 1.02); ax.grid(True, which='major', linestyle='--', alpha=0.4)
axes[0].set_ylabel('Test F1')
axes[0].legend(loc='lower right', fontsize=6.5)
fig.suptitle('Cross-Dataset QCNN–Classical Gap (N ≤ 5,000)', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_cross_dataset.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_cross_dataset.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig4_cross_dataset.pdf / .png (신규, 2026-08)')

# ============================================================
# Figure 5 — Large-N extension, ALL THREE datasets (U_TTN & U_5 vs classical, N up to 17,534)
# 2026-08-11: NSL-KDD 전용에서 3데이터셋 패널로 재구성 — 대N 효과 자체가
# 데이터셋 의존적이라는 게 핵심 발견이 되어, 세 데이터셋을 나란히 보여줘야
# 그 대비(NSL-KDD만 격차 좁혀짐, ToN_IoT 제자리, CICIDS-2017 격차 확대)가 한눈에 드러남.
# ============================================================
large_N_values = N_values + [10000, 17534]

def agg_any_n(records, model, metric):
    """agg()와 달리 전역 N_values가 아니라 데이터 자체에 있는 N값을 기준으로 집계."""
    result = {}
    for N in sorted(set(r['N'] for r in records)):
        vals = [r[model][metric] for r in records if r['N'] == N and model in r]
        if vals:
            result[N] = (np.mean(vals), np.std(vals))
    return result

def agg_extended(base_data, large_data, model, metric):
    result = agg_any_n(base_data, model, metric)
    extra = agg_any_n(large_data, model, metric)
    result.update(extra)
    return result

largeN_sets = {
    'NSL-KDD': {
        'base_uttn': nslkdd_ansatz_data['U_TTN'], 'base_u5': nslkdd_ansatz_data['U_5'],
        'large_uttn': largeN_uttn, 'large_u5': largeN_u5,
    },
    'ToN_IoT': {
        'base_uttn': toniot_ansatz_data['U_TTN'], 'base_u5': toniot_ansatz_data['U_5'],
        'large_uttn': load_json(os.path.join(SCRIPT_DIR, 'results_toniot_uttn_largeN', 'results_final.json')),
        'large_u5':   load_json(os.path.join(SCRIPT_DIR, 'results_toniot_u5_largeN', 'results_final.json')),
    },
    'CICIDS-2017': {
        'base_uttn': cicids_ansatz_data['U_TTN'], 'base_u5': cicids_ansatz_data['U_5'],
        'large_uttn': load_json(os.path.join(SCRIPT_DIR, 'results_cicids_uttn_largeN', 'results_final.json')),
        'large_u5':   load_json(os.path.join(SCRIPT_DIR, 'results_cicids_u5_largeN', 'results_final.json')),
    },
}

NsX = np.array(sorted(large_N_values))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, (ds_name, d) in zip(axes, largeN_sets.items()):
    uttn_f1_ext = agg_extended(d['base_uttn'], d['large_uttn'], 'QCNN', 'f1')
    u5_f1_ext   = agg_extended(d['base_u5'],   d['large_u5'],   'QCNN', 'f1')
    cnn_f1_ext  = agg_extended(d['base_uttn'], d['large_uttn'], 'CNN',  'f1')
    svm_f1_ext  = agg_extended(d['base_uttn'], d['large_uttn'], 'SVM',  'f1')

    mu_uttn = np.array([uttn_f1_ext[n][0] for n in NsX]); sd_uttn = np.array([uttn_f1_ext[n][1] for n in NsX])
    mu_u5   = np.array([u5_f1_ext[n][0] for n in NsX]);   sd_u5   = np.array([u5_f1_ext[n][1] for n in NsX])
    mu_cnn_x = np.array([cnn_f1_ext[n][0] for n in NsX])
    mu_svm_x = np.array([svm_f1_ext[n][0] for n in NsX])

    ax.plot(NsX, mu_cnn_x, color='#d62728', marker='s', linestyle='--', markersize=4, alpha=0.7, label='CNN-1D')
    ax.plot(NsX, mu_svm_x, color='#2ca02c', marker='^', linestyle=':', markersize=4, alpha=0.7, label='SVM')
    errorbar_dashed(ax, NsX, mu_uttn, sd_uttn, '#1f77b4', 'o', '-', 'U_TTN (12p)', markersize=4)
    errorbar_dashed(ax, NsX, mu_u5, sd_u5, '#17becf', 'X', '-', 'U_5 (36p)', markersize=4)
    ax.axvline(17534, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xscale('log'); ax.set_xticks(large_N_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.set_xlabel('Training Set Size $N$')
    ax.set_title(ds_name)
    ax.set_ylim(0, 1.02); ax.grid(True, which='major', linestyle='--', alpha=0.4)
axes[0].set_ylabel('Test F1')
axes[0].text(17534, 0.04, 'N=17,534\n(Gong et al. [9])', fontsize=6, ha='right', va='bottom', color='gray')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.suptitle('Large-$N$ Extension: Gap-Closing Is Dataset-Dependent', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_largeN.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_largeN.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig5_largeN.pdf / .png (3데이터셋 패널로 재구성, 2026-08-11)')

print('Done.')
