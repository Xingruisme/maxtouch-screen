"""BER vs bit rate, OOK and FSK, before (cc4 code bits) and after FEC (cc4 info bits); paper_fig_tablet_ber style.
Zero errors -> open marker at the 95 % Clopper-Pearson upper bound with a down arrow."""
import json, csv, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import beta
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
S = json.load(open('summary.json'))['summary']
STYLE = {"ook": dict(color="#1f5fbf", marker="o", label="OOK"), "fsk": dict(color="#00897b", marker="s", label="FSK")}
def cp(k, n):
    return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1), 1.0 if k == n else beta.ppf(0.975, k + 1, n - k))
panels = [("Before FEC (cc4 code bits)", "pre_fec_errors", "pre_fec_bits"), ("After FEC (cc4 decoded info bits)", "post_fec_errors", "post_fec_bits")]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
rows = []
for ax, (title, ke, kn) in zip(axs, panels):
    for mod, st in STYLE.items():
        rr = sorted([x for x in S if x['modulation'] == mod and x['decoder'] == 'cc4'], key=lambda x: x['baud'])
        b = np.array([x['baud'] for x in rr]); e = np.array([x[ke] for x in rr]); n = np.array([x[kn] for x in rr])
        ber = e / n; ci = np.array([cp(k, m) for k, m in zip(e, n)])
        y = np.where(e > 0, ber, ci[:, 1])
        dx = -0.04 if mod == 'ook' else 0.04
        ax.plot(b + dx, y, '-', color=st['color'], lw=1.8, label=st['label'])
        for xi, yi, ei, lo, hi in zip(b + dx, y, e, ci[:, 0], ci[:, 1]):
            if ei > 0:
                ax.errorbar(xi, yi, yerr=[[yi - lo], [hi - yi]], color=st['color'], marker=st['marker'], ms=7, capsize=3, lw=1.2)
            else:
                ax.plot(xi, yi, marker=st['marker'], ms=7, mfc='white', mec=st['color'], mew=1.6, color=st['color'])
                ax.annotate('', xy=(xi, yi / 2.2), xytext=(xi, yi), arrowprops=dict(arrowstyle='->', color=st['color'], lw=1.2))
        for x in rr:
            rows.append(dict(panel=title, modulation=mod, baud=x['baud'], errors=x[ke], bits=x[kn]))
    ax.set_title(title, fontsize=12)
    ax.set_xscale('linear'); ax.set_xticks([2, 3, 4]); ax.set_xlim(1.6, 4.4)
    ax.set_xlabel('bit rate on air [bps]', fontsize=11)
    ax.set_yscale('log'); ax.set_ylim(1e-4, 0.3)
    ax.grid(axis='y', which='major', color='#d9d9d9', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('BER', fontsize=11)
axs[0].legend(frameon=False, fontsize=10, loc='upper right')
axs[1].text(2.05, 0.11, 'OOK 2 bps: all errors from\nrecording 1 of 18 (Pad map froze)', fontsize=8, color='#444')
fig.suptitle('OnePlus Pad through body (ankle to finger, 1 mA, 37.4 kHz; FSK 0 = 445 kHz), trx frames, cc4\n'
             'open marker + arrow = 0 errors (drawn at the 95 % upper bound); bars = 95 % Clopper-Pearson', fontsize=9.5)
fig.tight_layout()
for e in ('png', 'pdf', 'svg'): fig.savefig(f'ber_vs_bitrate_pad.{e}', dpi=200)
with open('ber_vs_bitrate_pad.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
