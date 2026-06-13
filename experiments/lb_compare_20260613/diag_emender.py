import sys, os, subprocess, re
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import cmaes_search_v2 as C
C.DATA_PATH='/home/erikg/elman/data/pile.txt'; C.CHUNK_SIZE=2048; C.TOKENIZER_NAME='p50k_base'
params=dict(dim=2432,n_heads=212,n_state=32,depth=10,mixture_nonlin=0.9707993613680964,
            lr=0.0011443458778126467,batch_size=2)
out='experiments/lb_compare_20260613/runs_diag/Emender-mix'; os.makedirs(out,exist_ok=True)
cmd,_=C.build_train_command(params,'emender',15.0,out)
cmd=list(cmd)+['--heldout_tensor','experiments/lb_compare_20260613/heldout_p50k_2048.pt','--final_heldout_eval']
env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']='0'; env['HELDOUT_EVAL_BS']='8'; env['HELDOUT_REPORT_NONAVG']='1'
env.setdefault('XMA_PATH','/home/erikg/xma')
logp=out+'/diag.log'
with open(logp,'w') as lf:
    lf.write('CMD '+' '.join(cmd)+'\n\n'); lf.flush()
    p=subprocess.run(cmd,env=env,stdout=lf,stderr=subprocess.STDOUT)
txt=open(logp).read()
for k in ['FINAL_HELDOUT_CE_NONAVG','FINAL_HELDOUT_BPB_NONAVG','FINAL_HELDOUT_CE:','FINAL_HELDOUT_BPB:','FINAL_LOSS_LAST100']:
    m=re.findall(re.escape(k)+r' *([\d.]+)',txt)
    print(k, m[-1] if m else 'NONE')
print('rc',p.returncode)
