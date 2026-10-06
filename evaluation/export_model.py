"""Export a self-describing A3 context checkpoint for the isolated evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def export(checkpoint, output, skip_unchanged=False):
    checkpoint, output = Path(checkpoint), Path(output)
    manifest = output.with_suffix('.json')
    stat = checkpoint.stat()
    identity = [stat.st_mtime_ns, stat.st_size]
    if skip_unchanged and manifest.exists() and output.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('checkpoint_identity') == identity:
            return dict(previous, changed=False)
    import torch
    import onnx
    from rlcard.agents.dmc_agent.model import DMCModel
    torch.set_num_threads(1)
    # The trainer publishes by rename. Loading one open inode gives one complete
    # generation even if another checkpoint is published during this export.
    with checkpoint.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        payload = torch.load(stream, map_location='cpu', weights_only=True)
    spec = payload['model_spec']
    if spec.get('architecture') != 'context' or spec.get('feature_schema') != 'a3-v12-lite-v1':
        raise ValueError('Exporter requires the A3 context feature schema')
    if not spec.get('share_weights'):
        raise ValueError('This evaluator requires shared role weights')
    fields = ('state_shape', 'action_shape', 'mlp_layers', 'architecture', 'aux_classes',
              'share_weights', 'history_encoder', 'history_steps')
    model = DMCModel(**{key: spec[key] for key in fields}, device='cpu')
    states = payload['model_state_dict']
    if any(set(state) != set(states[0]) or any(not torch.equal(state[k], states[0][k]) for k in states[0]) for state in states[1:]):
        raise ValueError('Checkpoint role weights are inconsistent')
    model.get_agent(0).load_state_dict(states[0]); model.eval()
    net = model.get_agent(0).net
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Non-finite checkpoint weights')

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.net = net
        def forward(self, obs_static, hist_tokens, action_feat, state_indices):
            obs = torch.cat((obs_static, hist_tokens.flatten(1)), 1)
            context, _ = self.net.encode_state(obs)
            play = self.net._play_q(context.index_select(0, state_indices), action_feat)
            auxiliary = (self.net.aux_head(context) if self.net.aux_head is not None else
                         context.new_zeros((context.shape[0], 17)))
            return play, auxiliary, self.net.declare_head(context)

    wrapper = Export().eval()
    example = (torch.zeros(2,556), torch.zeros(2,24,88), torch.zeros(3,111), torch.tensor([0,0,1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f'.{os.getpid()}.tmp.onnx')
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(wrapper, example, str(temporary), input_names=['obs_static','hist_tokens','action_feat','state_indices'],
                output_names=['play_q','aux_logits','declare_q'], opset_version=17,
                dynamic_axes={'obs_static':{0:'states'},'hist_tokens':{0:'states'},
                              'action_feat':{0:'candidates'},'state_indices':{0:'candidates'},
                              'play_q':{0:'candidates'},'aux_logits':{0:'states'},'declare_q':{0:'states'}},
                dynamo=False, external_data=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(temporary))
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
    import numpy as np
    maximum = 0.
    for states_count, count in ((1,1), (3,7), (8,23)):
        torch.manual_seed(100+count)
        inputs = (torch.randint(0,2,(states_count,556)).float(), torch.randint(0,2,(states_count,24,88)).float(),
                  torch.randint(0,2,(count,111)).float(), torch.arange(count)%states_count)
        if count == 1: inputs[1].zero_()
        with torch.no_grad(): expected = wrapper(*inputs)
        actual = session.run(None,
            {name:value.numpy() for name,value in zip(['obs_static','hist_tokens','action_feat','state_indices'],inputs)})
        for left, right in zip(actual,expected):
            np.testing.assert_allclose(left,right.numpy(),rtol=2e-4,atol=3e-5)
            maximum=max(maximum,float(np.abs(left-right.numpy()).max()))
    del session
    os.replace(temporary, output)
    metadata = dict(layout='batched-context-v1', frames=payload['frames'], model_spec=spec,
                    torch_version=str(torch.__version__), onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                    checkpoint_identity=[stat.st_mtime_ns,stat.st_size], sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    size_bytes=output.stat().st_size, onnx_max_abs_error=maximum)
    temp_json=manifest.with_suffix(f'.{os.getpid()}.tmp.json');temp_json.write_text(json.dumps(metadata,indent=2));os.replace(temp_json,manifest)
    return dict(metadata,changed=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--if-changed',action='store_true')
    parser.add_argument('--initial-checkpoint')
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint)
    if not checkpoint.exists() and args.initial_checkpoint:
        checkpoint=Path(args.initial_checkpoint)
    if not checkpoint.exists():
        print(json.dumps({'waiting':True}));return
    print(json.dumps(export(checkpoint,args.output,args.if_changed)))


if __name__=='__main__':main()
