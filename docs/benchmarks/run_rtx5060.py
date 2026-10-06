"""Linux single-GPU reproduction script; run from a prepared CUDA environment."""
import csv
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
repo = Path(__file__).resolve().parents[2]
frames = int(sys.argv[1]) if len(sys.argv) > 1 else 64000
output = repo / 'experiments' / ('rtx5060-profiles-' + str(frames))
output.mkdir(parents=True, exist_ok=True)
profiles = [('cpu_actor_32', True, 32), ('gpu_actor_32', False, 32), ('gpu_actor_128', False, 128)]
report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
              device=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
              frames=frames, batch_size=32, unroll_length=20, model='ResNet [512]*5, A3 auxiliary heads',
              backend='cpp', shared_weights=True,
              methodology='End-to-end child process time including imports, spawn, training and final checkpoint; whole-device memory sampled every 0.5s; two seeds per profile.',
              runs=[])

def gpu_sample():
    values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(',')
    return [int(x.strip()) for x in values]

for label, cpu_actor, envs in profiles:
    for repeat in range(2):
        xpid=f'{label}_seed{42+repeat}'
        args=[sys.executable,'-m','examples.run_dmc','--env','a3dizhu','--backend','cpp',
              '--cuda','0','--training_device','0','--num_actors','1','--envs_per_actor',str(envs),
              '--total_frames',str(frames),'--batch_size','32','--unroll_length','20','--num_buffers','64',
              '--hidden_sizes','512','512','512','512','512','--share_weights','--seed',str(42+repeat),
              '--xpid',xpid,'--savedir',str(output)]
        if cpu_actor:args.append('--actor_on_cpu')
        initial=gpu_sample()
        samples=[]
        started=time.perf_counter()
        with (output/(xpid+'.log')).open('w') as logfile:
            proc=subprocess.Popen(args,cwd=repo,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                samples.append(gpu_sample())
                if time.perf_counter()-started>240:
                    os.killpg(proc.pid,signal.SIGINT)
                    proc.wait(timeout=20)
                    break
                time.sleep(.5)
            code=proc.wait()
        elapsed=time.perf_counter()-started
        result=dict(profile=label,seed=42+repeat,exit_code=code,elapsed_seconds=elapsed,
                    baseline_gpu_mib=initial[0],peak_total_gpu_mib=max(x[0] for x in samples),
                    average_sampled_gpu_utilization=sum(x[1] for x in samples)/len(samples))
        csvpath=output/xpid/'logs.csv'
        if csvpath.exists():
            with csvpath.open() as stream:
                fields=stream.readline().lstrip('# ').strip().split(',')
                rows=list(csv.DictReader((line for line in stream if line.strip() and not line.startswith('#')),fieldnames=fields))
            result['stats']=rows[-1]
            result['end_to_end_samples_per_second']=int(rows[-1]['frames'])/elapsed
        result['cuda_ipc_warning']='CudaIPCTypes' in (output/(xpid+'.log')).read_text()
        report['runs'].append(result)
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(result),flush=True)
        if code!=0:raise SystemExit(code)
print('REPORT',output/'report.json',flush=True)
