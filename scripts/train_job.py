"""Run a finite A3 training job with a private MPS server and durable status."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.track_training import atomic_json


def last_frames(path):
    if not path.exists(): return 0
    with path.open('rb') as stream:
        fields=stream.readline().decode().lstrip('# ').strip().split(',')
        stream.seek(0,2); size=stream.tell();stream.seek(max(0,size-8192))
        rows=[line for line in stream.read().decode().splitlines() if line and not line.startswith('#')]
    if not rows:return 0
    for row in reversed(rows):
        values=next(csv.reader([row]))
        if len(values)!=len(fields):continue
        try:return int(float(dict(zip(fields,values)).get('frames',0)))
        except ValueError:continue
    return 0


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=3_000_000_000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if os.name!='posix':parser.error('The MPS job runner requires Linux')
    if args.frames<=0:parser.error('frames must be positive')
    directory=args.run_dir.resolve();directory.mkdir(parents=True,exist_ok=True)
    if (directory/'model.tar').exists() or (directory/'initial_policy.tar').exists():
        raise FileExistsError('Use a new run directory to avoid overwriting training data')
    started=datetime.now(timezone.utc).isoformat()
    status=dict(state='starting',started_at=started,target_frames=args.frames,seed=args.seed,pid=os.getpid())
    atomic_json(directory/'status.json',status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PATH='/usr/sbin:'+os.environ.get('PATH',''),PYTHONPATH=str(ROOT))
    sysroot=ROOT.parent/'sysroot/usr/include'
    if sysroot.exists():env['CPATH']=str(sysroot/f'python{sys.version_info.major}.{sys.version_info.minor}')+':'+str(sysroot)
    command=[sys.executable,'-u','-m','examples.run_dmc','--env','a3dizhu-v12','--backend','cpp',
             '--cuda','0','--training_device','0','--num_actors','2','--actor_threads','2',
             '--envs_per_actor','512','--batch_size','32','--unroll_length','20','--num_buffers','64',
             '--share_weights','--history_steps','24','--precision','bf16','--actor_half_weights',
             '--shared_transport','--compile_learner','--compile_optimizer','--cuda_graph_learner','--compile_actor',
             '--initial_epsilon','.08','--final_epsilon','.01','--total_frames',str(args.frames),
             '--seed',str(args.seed),'--save_interval','5','--savedir',str(directory.parent),'--xpid',directory.name]
    atomic_json(directory/'job.json',dict(command=command,target_frames=args.frames,seed=args.seed))
    # A separate inference-only initial checkpoint supplies a genuine frame-zero
    # evaluation without pretending to be a resumable optimizer checkpoint.
    import torch
    import rlcard
    from rlcard.agents.dmc_agent import DMCTrainer
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    initial_trainer=DMCTrainer(rlcard.make('a3dizhu-v12',config={'reward_mode':'game','seed':args.seed}),
        cuda='',share_weights=True,history_steps=24,total_frames=args.frames,seed=args.seed)
    initial=initial_trainer.model_func('cpu')
    torch.save(dict(model_spec=initial_trainer._model_spec(),frames=0,share_weights=True,
                    model_state_dict=[a.state_dict() for a in initial.get_agents()]),directory/'initial_policy.tar')
    del initial,initial_trainer
    process=None; interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
        if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with tempfile.TemporaryDirectory(prefix='rlcard-mps-') as pipe:
        logs=directory/'mps-log';logs.mkdir(exist_ok=True)
        env.update(CUDA_MPS_PIPE_DIRECTORY=pipe,CUDA_MPS_LOG_DIRECTORY=str(logs))
        mps_started=False
        try:
            subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
            mps_started=True
            with (directory/'training.log').open('w') as logfile:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                status.update(state='running',training_pid=process.pid,mps_pipe=pipe)
                atomic_json(directory/'status.json',status)
                last_count=0;last_time=time.monotonic()
                while True:
                    try:code=process.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        current=max(last_count,last_frames(directory/'logs.csv'))
                        now=time.monotonic()
                        status.update(frames=current,samples_per_second=(current-last_count)/(now-last_time),
                                      updated_at=datetime.now(timezone.utc).isoformat())
                        atomic_json(directory/'status.json',status)
                        last_count,last_time=current,now
            frames=last_frames(directory/'logs.csv')
            status.update(state='completed' if code==0 and frames>=args.frames else 'interrupted' if interrupted else 'failed',
                          returncode=code,frames=frames,finished_at=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            status.update(state='failed',error=str(error),finished_at=datetime.now(timezone.utc).isoformat())
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            if mps_started:
                result=subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',env=env,capture_output=True,text=True,timeout=30)
                status['mps_stopped']=result.returncode==0
            atomic_json(directory/'status.json',status)


if __name__=='__main__':main()
