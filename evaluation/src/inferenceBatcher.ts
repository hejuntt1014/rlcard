import * as ort from 'onnxruntime-node';

export interface InferenceResult { q: Float32Array; auxiliary: Float32Array; declaration: Float32Array; }
interface Request {
  obs: Float32Array; history: Float32Array; actions: Float32Array;
  resolve: (value: InferenceResult) => void; reject: (error: unknown) => void;
}

/** Coalesce independent games; at most one native ORT call is active per worker. */
export class InferenceBatcher {
  private queue: Request[] = [];
  private scheduled = false;
  private running = false;
  constructor(private session: () => Promise<ort.InferenceSession>) {}
  infer(obs: Float32Array, history: Float32Array, actions: Float32Array): Promise<InferenceResult> {
    return new Promise((resolve, reject) => {
      this.queue.push({ obs, history, actions, resolve, reject }); this.schedule();
    });
  }
  private schedule() {
    if (!this.scheduled && !this.running && this.queue.length) {
      this.scheduled = true; setImmediate(() => void this.flush());
    }
  }
  private async flush() {
    this.scheduled = false; this.running = true;
    const requests = this.queue.splice(0, 64);
    try {
      const count = requests.length;
      const candidates = requests.reduce((sum, r) => sum + r.actions.length / 111, 0);
      const obs = new Float32Array(count * 556), hist = new Float32Array(count * 24 * 88);
      const actions = new Float32Array(candidates * 111), indices = new BigInt64Array(candidates);
      let cursor = 0;
      requests.forEach((r, i) => {
        obs.set(r.obs, i*556); hist.set(r.history, i*24*88); actions.set(r.actions, cursor*111);
        indices.fill(BigInt(i), cursor, cursor+r.actions.length/111); cursor += r.actions.length/111;
      });
      const result = await (await this.session()).run({
        obs_static: new ort.Tensor('float32', obs, [count,556]),
        hist_tokens: new ort.Tensor('float32', hist, [count,24,88]),
        action_feat: new ort.Tensor('float32', actions, [candidates,111]),
        state_indices: new ort.Tensor('int64', indices, [candidates]),
      });
      cursor = 0;
      requests.forEach((r, i) => {
        const n = r.actions.length/111;
        r.resolve({ q: (result.play_q!.data as Float32Array).slice(cursor, cursor+n),
          auxiliary: (result.aux_logits!.data as Float32Array).slice(i*17, (i+1)*17),
          declaration: (result.declare_q!.data as Float32Array).slice(i*2, (i+1)*2) });
        cursor += n;
      });
    } catch (error) { requests.forEach(r => r.reject(error)); }
    finally { this.running = false; this.schedule(); }
  }
}
