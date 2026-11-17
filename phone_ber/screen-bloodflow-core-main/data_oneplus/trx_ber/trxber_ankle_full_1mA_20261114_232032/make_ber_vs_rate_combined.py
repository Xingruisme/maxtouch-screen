"""BER vs bit rate, OOK and FSK, before (dashed, cc4 code bits) and after FEC (solid, cc4 info bits) in one panel.
Zero errors -> open marker at the 95 % Clopper-Pearson upper bound with a down arrow."""
import json, csv, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
S = json.load(open('summary.json'))['summary']
STYLE = {"ook": dict(color="#1f5fbf", marker="o", label="OOK"), "fsk": dict(color="#00897b", marker="s", label="FSK")}
KIND = {"before": dict(ke="pre_fec_errors", kn="pre_fec_bits", ls="--", lw=1.5, filled=False, dx=-0.06),
        "after": dict(ke="post_fec_errors", kn="post_fec_bits", ls="-", lw=2.0, filled=True, dx=0.06)}
def cp(k, n):
    return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1), 1.0 if k == n else beta.ppf(0.975, k + 1, n - k))
fig, ax = plt.subplots(figsize=(7.4, 5.2))
rows = []
for mod, st in STYLE.items():
    rr = sorted([x for x in S if x['modulation'] == mod and x['decoder'] == 'cc4'], key=lambda x: x['baud'])
    for kind, k in KIND.items():
        b = np.array([x['baud'] for x in rr]); e = np.array([x[k['ke']] for x in rr]); n = np.array([x[k['kn']] for x in rr])
        ci = np.array([cp(a, m) for a, m in zip(e, n)]); y = np.where(e > 0, e / n, ci[:, 1])
        x = b + k['dx'] + (-0.015 if mod == 'ook' else 0.015)
        ax.plot(x, y, k['ls'], color=st['color'], lw=k['lw'])
        for xi, yi, ei, lo, hi in zip(x, y, e, ci[:, 0], ci[:, 1]):
            if ei > 0:
                ax.errorbar(xi, yi, yerr=[[yi - lo], [hi - yi]], color=st['color'], marker=st['marker'], ms=7, capsize=3,
                            lw=1.0, mfc=st['color'] if k['filled'] else 'white', mec=st['color'], mew=1.5)
            else:
                ax.plot(xi, yi, marker=st['marker'], ms=7, mfc='white', mec=st['color'], mew=1.5)
                ax.annotate('', xy=(xi, yi / 2.2), xytext=(xi, yi), arrowprops=dict(arrowstyle='->', color=st['color'], lw=1.1))
        for xi, yi, ei, ni in zip(x, y, e, n):           # errors / bits counted, beside each marker
            ax.annotate(f"{ei}/{ni}", (xi, yi), xytext=(-12 if kind == 'before' else 8, -11 if mod == 'fsk' else 7),
                        textcoords='offset points', ha='right' if kind == 'before' else 'left', fontsize=7.5, color=st['color'])
        for xr, ei, ni in zip(b, e, n):
            rows.append(dict(modulation=mod, fec=kind, bit_rate_bps=xr, errors=int(ei), bits=int(ni), ber=ei / ni))
ax.set_xticks([2, 3, 4]); ax.set_xlim(1.6, 4.4); ax.set_xlabel('bit rate on air [bps]', fontsize=12)
ax.set_yscale('log'); ax.set_ylim(1e-4, 0.3); ax.set_ylabel('BER', fontsize=12)
ax.grid(axis='y', which='major', color='#d9d9d9', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
h = [Line2D([], [], color=STYLE['ook']['color'], marker='o', lw=2, label='OOK'),
     Line2D([], [], color=STYLE['fsk']['color'], marker='s', lw=2, label='FSK'),
     Line2D([], [], color='#555', ls='--', lw=1.5, label='before FEC (code bits)'),
     Line2D([], [], color='#555', ls='-', lw=2.0, label='after FEC (decoded info bits)')]
ax.legend(handles=h, frameon=False, fontsize=9.5, loc='upper right')
fig.text(0.5, 0.02, '6,912 bits transmitted (OOK 3,456, FSK 3,456)', fontsize=9, color='#333', ha='center')
ax.set_title('OnePlus Pad through-body BER (1 mA)', fontsize=12)
fig.tight_layout(rect=(0, 0.05, 1, 1))
for e in ('png', 'pdf', 'svg'): fig.savefig(f'ber_vs_bitrate_pad_combined.{e}', dpi=200)
with open('ber_vs_bitrate_pad_combined.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
