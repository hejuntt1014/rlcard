"""Evaluate immutable checkpoint exports with bounded CPU concurrency."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(5):
        try: os.replace(temporary, path); return
        except PermissionError:
            if attempt == 4: raise
            time.sleep(.05)


@contextmanager
def tracker_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error: raise RuntimeError('A tracker already owns this result directory') from error
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def remote_export(args, destination):
    checkpoint = args.run_dir.rstrip('/') + '/model.tar'
    remote_model = args.run_dir.rstrip('/') + '/evaluation/model.onnx'
    command = ['env','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',args.remote_python,
               '-m','evaluation.export_model','--checkpoint',checkpoint,'--output',remote_model,
               '--initial-checkpoint',args.run_dir.rstrip('/')+'/initial_policy.tar','--if-changed']
    shell = 'cd ' + shlex.quote(args.remote_repo) + ' && ' + shlex.join(command)
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.ssh_host,shell],
                            capture_output=True,text=True,encoding='utf-8',timeout=180)
    if result.returncode: raise RuntimeError(result.stderr[-4000:])
    metadata=json.loads(result.stdout.strip().splitlines()[-1])
    if metadata.get('waiting'): return None, metadata
    local_model=destination/f"model-{metadata['frames']}-{metadata['sha256'][:12]}.onnx"
    if not local_model.exists():
        temporary=local_model.with_suffix('.download')
        subprocess.run(['scp','-q','-o','BatchMode=yes',f'{args.ssh_host}:{remote_model}',str(temporary)],
                       check=True,timeout=120)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=metadata['sha256']:
            raise RuntimeError('Downloaded model checksum mismatch')
        os.replace(temporary,local_model)
    atomic_json(local_model.with_suffix('.json'),metadata)
    return local_model,metadata


def terminate_evaluation(process):
    if process.poll() is not None: return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
    else:
        os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: process.kill();process.wait()


def benchmark(model, args, output):
    command=[shutil.which('node') or 'node','--import','tsx','src/cli.ts',
             '--model',str(model),'--games',str(args.games),'--workers',str(args.workers),
             '--onnx-threads',str(args.onnx_threads),'--batch-games',str(args.batch_games),
             '--seed',str(args.seed),'--output',str(output)]
    options = ({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS}
               if os.name=='nt' else {'start_new_session':True})
    with output.with_suffix('.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,**options)
        try:
            code=process.wait(timeout=args.timeout)
            if code not in (0,2): raise RuntimeError(f'Evaluation failed ({code}); see {output.with_suffix(".log")}')
            result=json.loads(output.read_text(encoding='utf-8'))
            if result['games'] != args.games: raise RuntimeError('Incomplete evaluation')
            return result
        finally: terminate_evaluation(process)


def start_dashboard(directory, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            if self.path=='/api/status':
                path=directory/'status.json'; kind='application/json'
            elif self.path=='/api/metrics':
                path=directory/'metrics.json'; kind='application/json'
            elif self.path in ('/','/index.html'):
                path=ROOT/'dashboard.html'; kind='text/html; charset=utf-8'
            else: self.send_error(404);return
            data=path.read_bytes() if path.exists() else b'{}'
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


def main():
    config=json.loads((ROOT/'local.json').read_text()) if (ROOT/'local.json').exists() else {}
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--games',type=int,default=2000)
    parser.add_argument('--interval',type=float,default=5,help='Minutes between evaluation checks')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--onnx-threads',type=int,default=1)
    parser.add_argument('--batch-games',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--model',type=Path,help='Evaluate a local exported ONNX model')
    parser.add_argument('--ssh-host',default=config.get('ssh_host','pve'))
    parser.add_argument('--remote-repo',default=config.get('remote_repo'))
    parser.add_argument('--remote-python',default=config.get('remote_python'))
    parser.add_argument('--run-dir',default=config.get('run_dir'))
    parser.add_argument('--output-dir',type=Path,default=Path(config.get('output_dir',ROOT/'results/tracking')))
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--no-dashboard',action='store_true')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if min(args.games,args.interval,args.workers,args.onnx_threads,args.batch_games,args.timeout)<=0:
        parser.error('Counts, interval and timeout must be positive')
    if not args.model and not all((args.remote_repo,args.remote_python,args.run_dir)):
        parser.error('Supply --model or configure remote_repo, remote_python and run_dir in evaluation/local.json')
    directory=args.output_dir.resolve(); directory.mkdir(parents=True,exist_ok=True)
    artifacts=directory/'models';artifacts.mkdir(exist_ok=True)
    history_path=directory/'metrics.json'
    history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    sources=[p for p in sorted((ROOT/'src').rglob('*.ts'))+sorted((ROOT/'shared').rglob('*.ts')) if not p.name.endswith('.test.ts')]
    sources.append(ROOT/'package-lock.json')
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    protocol=dict(games=args.games,seed=args.seed,protocol='seeded-mix-standard-v1',evaluator_sha256=source_hash)
    def status(state,**details):
        value=dict(state=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                   workers=args.workers,onnx_threads=args.onnx_threads,games=args.games,interval_minutes=args.interval,**details)
        atomic_json(directory/'status.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    with tracker_lock(directory/'tracker.lock'):
        server=None
        try:
            if not args.no_dashboard:
                server=start_dashboard(directory,args.port)
                if not args.no_browser: webbrowser.open(f'http://127.0.0.1:{args.port}')
            while True:
                started=time.monotonic()
                try:
                    status('exporting')
                    if args.model:
                        model=args.model.resolve();metadata=json.loads(model.with_suffix('.json').read_text())
                    else: model,metadata=remote_export(args,artifacts)
                    if model is None: status('waiting_checkpoint')
                    elif history and history[-1].get('model',{}).get('sha256')==metadata['sha256'] and history[-1].get('protocol')==protocol:
                        status('idle' if history[-1]['evaluation']['valid'] else 'invalid_evaluation',
                               frames=metadata['frames'],message='Checkpoint already evaluated',latest=history[-1]['evaluation'])
                    else:
                        status('evaluating',frames=metadata['frames'])
                        result=benchmark(model,args,directory/f"evaluation-{metadata['frames']}.json")
                        entry=dict(timestamp=datetime.now(timezone.utc).isoformat(),model=metadata,protocol=protocol,evaluation=result)
                        history.append(entry);atomic_json(history_path,history)
                        status('idle' if result['valid'] else 'invalid_evaluation',frames=metadata['frames'],latest=result)
                except KeyboardInterrupt: raise
                except Exception as error:
                    status('error',error=str(error))
                    if args.once: raise
                if args.once: break
                time.sleep(max(1,args.interval*60-(time.monotonic()-started)))
        except KeyboardInterrupt: status('stopped')
        finally:
            if server: server.shutdown();server.server_close()


if __name__=='__main__':main()
