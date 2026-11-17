import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0,'/home/semanticsignals/xingru/screen-bloodflow-core-main'); import phone_body_sweep as PS
S=str(Path(__file__).resolve().parent)+'/'
X=Path.home()/'xingru'
p4=json.load(open(S+'px4_shift.json')); p6=json.load(open(S+'px6_lock_all.json')); pad=json.load(open(S+'pad_update.json'))
GREY='#9a9a9a'
fig,ax=plt.subplots(4,1,figsize=(11,11),sharex=True,gridspec_kw=dict(height_ratios=[1,1.2,1,0.8]))
def lab(a,xs,ys,dy=6,fmt='{:.1f}k',skip=()):
    for x,y in zip(xs,ys):
        if any(abs(x-s)/s<0.01 for s in skip): continue
        a.annotate(fmt.format(x/1e3),(x,y),xytext=(0,dy if y>=0 else -13),textcoords='offset points',ha='center',fontsize=7.5)
# Pixel 4
a=ax[0]; c='#c0602a'; f=np.array([h[0] for h in p4['hand']]); y=np.array([h[1] for h in p4['hand']])
a.plot([h[0] for h in p4['control']],[h[1] for h in p4['control']],'-',color=GREY,lw=1.2,label='control, no TX')
a.plot(f,y,'-',color=c,lw=1.1); r=y>2; a.plot(f[~r],y[~r],'o',ms=3.5,color=c,mfc='white'); a.plot(f[r],y[r],'o',ms=7,color=c,label='through body, 0.1 mA (response)')
lab(a,f[r],y[r]); a.set_ylabel('panel shift\nON - OFF [counts]'); a.set_ylim(-1.5,8.5)
a.set_title('Pixel 4 (STM FTS 15 x 32): 7 narrow responses',loc='left',fontsize=10)
# Pixel 6
a=ax[1]; c='#2a6fdb'; H=p6['hand_1k_500k_20261114_202504']; C=p6['ctl_20261114_204154']
f=np.array([o['f'] for o in H]); y=np.array([o['shift'] for o in H]); r=np.array([o['flag']=='LOCK' for o in H])
a.plot([o['f'] for o in C],[o['shift'] for o in C],'-',color=GREY,lw=1.2,label='control, no TX')
a.plot(f,y,'-',color=c,lw=1.1); a.plot(f[~r],y[~r],'o',ms=3.5,color=c,mfc='white'); a.plot(f[r],y[r],'o',ms=7,color=c,label='through body, 0.1 mA (response)')
lab(a,f[r],y[r]); a.axvspan(950,1550,color='#eee',lw=0)
a.set_ylabel('panel shift\nON - OFF [counts]'); a.set_ylim(-12,13)
a.set_title('Pixel 6 (STM FTM5 16 x 34): 7 narrow responses (2 negative steps)',loc='left',fontsize=10)
# Pad
a=ax[2]; c='#2e8b57'; H=pad['hand_1k_500k_20261114_211114']; C=pad['ctl_20261114_210534']
r=np.array([u['on'] for u in H])>=0.8                 # refreshed in >= 80 % of ON frames (response)
A=json.load(open(S+'pad_activity_rms.json')); HA=A['hand_1k_500k_20261114_211114']; CA=A['ctl_20261114_210534']
f=np.array([u['f'] for u in HA]); y=np.array([u['on'] for u in HA])
a.plot([u['f'] for u in CA],[u['on'] for u in CA],'-',color=GREY,lw=1.2,label='control, no TX')
a.plot(f,y,'-',color=c,lw=1.1); a.plot(f[~r],y[~r],'o',ms=3.5,color=c,mfc='white'); a.plot(f[r],y[r],'o',ms=6,color=c,label='through body, 0.1 mA (response)')
i=int(np.argmax(y)); a.annotate(f'{f[i]/1e3:.1f}k',(f[i],y[i]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7.5)
a.set_ylabel('map activity, ON\nRMS frame-to-frame\nchange [counts]'); a.set_ylim(-0.15,3.2)
a.set_title('OnePlus Pad (Novatek 38 x 54): broadband response 9.9-117 kHz and 189-368 kHz (map activity, no level shift)',loc='left',fontsize=10)
# 8 Pro
a=ax[3]; c='#8a4fbf'; d=X/'device-phone-oneplus8pro/data/body_sweep'
fc,nc,_,_=PS.response_curve(d/'ctl_20261114_162814'); fh,nh,bh,_=PS.response_curve(d/'hand1_20261114_160804')
k=fc>=900; a.plot(fc[k],nc[k],'-',color=GREY,lw=1.2,label='control, no TX')
a.plot(fh,nh,'o-',color=c,ms=6,lw=1.1,label='through body, 1 mA (run stopped after 4 carriers)')
i=int(np.argmax(nh)); a.annotate(f'{fh[i]/1e3:.0f}k',(fh[i],nh[i]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7.5)
a.set_ylabel('max noise rise\nON vs OFF [dB]'); a.set_ylim(0,14)
a.set_title('OnePlus 8 Pro (Samsung S6SY761 17 x 38): incomplete - only 20-26.6 kHz recorded; 22 kHz responded',loc='left',fontsize=10)
for a in ax:
    a.axvline(37404,color='#555',lw=.7,ls=':'); a.spines[['top','right']].set_visible(False); a.grid(axis='y',color='#eee')
    a.legend(loc='upper left',fontsize=7.5,frameon=False,ncol=2)
ax[0].text(37404*1.06,3.6,'37.4 kHz: Pixel 4, Pixel 6 and Pad',fontsize=7.5,color='#444')
b=ax[-1]; b.set_xscale('log'); b.set_xlim(900,560e3); b.set_xlabel('carrier frequency [Hz]')
b.set_xticks([1e3,2e3,5e3,1e4,2e4,5e4,1e5,2e5,5e5]); b.set_xticklabels(['1k','2k','5k','10k','20k','50k','100k','200k','500k'])
fig.suptitle('Through-body frequency sweep (hand to finger, TeensyTRX), 1-500 kHz, 10 % steps, quick mode - phones and tablet',fontsize=11.5)
fig.tight_layout()
out=X/'screen-bloodflow-core-main/data_body_sweep_all/response_vs_frequency_all_devices'
for e in ('png','pdf'): fig.savefig(f'{out}.{e}',dpi=130)
print(out)
