import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=json.load(open('summary.json'))['summary']
get=lambda mod,b,dec:[x for x in S if x['modulation']==mod and round(x['baud'])==b and x['decoder']==dec][0]
bars=[('uncoded (none_full)','#eb6834'),('cc4 before FEC (code bits)','#9cc0ef'),('cc4 after FEC (info bits)','#2a78d6')]
fig,ax=plt.subplots(figsize=(13,5.6)); w=0.26; xt=[];xl=[]
for gi,(mod,b) in enumerate([(m,b) for m in ('ook','fsk') for b in (2,3,4)]):
    x0=gi+(0.5 if mod=='fsk' else 0)
    nf=get(mod,b,'none_full'); cc=get(mod,b,'cc4')
    vals=[(nf['post_fec_errors'],nf['post_fec_bits']),(cc['pre_fec_errors'],cc['pre_fec_bits']),(cc['post_fec_errors'],cc['post_fec_bits'])]
    for bi,((e,n),(lab,col)) in enumerate(zip(vals,bars)):
        x=x0+(bi-1)*(w+0.02); y=100*e/n
        ax.bar(x,max(y,0.15),w,color=col if e else 'none',edgecolor=col,lw=1.5,label=lab if gi==0 else None)
        ax.text(x,max(y,0.15)+0.2,f"{e}/{n}\n"+("0 %" if e==0 else f"{y:.2g} %"),ha='center',va='bottom',fontsize=7.6)
    ax.text(x0,-1.9,f"frames OK\nuncoded {nf['frames']-nf['frame_errors']}/3, cc4 {cc['frames']-cc['frame_errors']}/3",ha='center',fontsize=7.6,color='#555')
    xt.append(x0); xl.append(f"{mod.upper()} {b} baud")
ax.set_xticks(xt); ax.set_xticklabels(xl); ax.set_ylim(-2.6,12); ax.axhline(0,color='#888',lw=.8)
ax.set_ylabel('bit error rate [%]'); ax.legend(frameon=False,fontsize=8.5,loc='upper right')
ax.annotate('all OOK 2-baud errors are from recording 1 of 18:\nthe Pad stopped refreshing in stretches (carrier on, map frozen).\nRecordings 2-3 alone: uncoded 2/192, cc4 0/384 -> 0/192',
            xy=(0.27,8.1),xytext=(0.6,9.6),fontsize=8,color='#444',arrowprops=dict(arrowstyle='->',color='#888'))
ax.set_title('OnePlus Pad, ankle to finger, 1 mA, 37.4 kHz (FSK 0 = 445 kHz): before vs after FEC, 3 frames of each per cell',fontsize=10.5)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('before_after_fec_pad.png',dpi=140); fig.savefig('before_after_fec_pad.pdf')
