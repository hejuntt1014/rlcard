"""Optional fixed-batch CUDA replay of a complete RMSprop learning update."""
import torch


class CUDALearner:
    """One graph per declaration phase; keep live weights, LR and RMSprop state.

    Warmup/capture changes are rolled back, so the first real batch is learned
    exactly once. Inputs and graphs are private to one learner policy. FP16
    dynamic scaling is deliberately excluded from this explicit replay path.
    """
    def __init__(self, agent, optimizer, objective, precision, max_grad_norm):
        if precision not in ('fp32', 'bf16'):
            raise ValueError('CUDA learner replay supports fp32 and bf16')
        if not isinstance(optimizer, torch.optim.RMSprop):
            raise ValueError('CUDA learner replay requires RMSprop')
        self.agent, self.optimizer, self.objective = agent, optimizer, objective
        self.precision, self.max_grad_norm = precision, max_grad_norm
        self.parameters = list(agent.parameters())
        self.device = self.parameters[0].device
        self.cache = {}
        # Allocate state once, including currently inactive heads. Their counters
        # remain zero until used; homogeneous phases still leave gradients absent.
        for group in optimizer.param_groups:
            if not group['capturable'] or not isinstance(group['lr'], torch.Tensor):
                raise ValueError('Replay requires capturable RMSprop and a device learning-rate tensor')
            for parameter in group['params']:
                state = optimizer.state[parameter]
                if not state:
                    state['step'] = torch.zeros((), device=parameter.device)
                    state['square_avg'] = torch.zeros_like(parameter)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(parameter)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(parameter)

    def _update(self, inputs, phase):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=self.precision == 'bf16',
                            dtype=torch.bfloat16, cache_enabled=False):
            loss = self.objective(inputs['state'].float(), inputs['action'].float(),
                                  inputs['target'], inputs.get('aux_target'), phase=phase)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()
        return loss

    def _capture(self, batch, phase):
        inputs = {key: value.flatten(0, 1).to(self.device).clone()
                  for key, value in batch.items()
                  if key in ('state', 'action', 'target', 'aux_target')}
        tensors = self.parameters + [value for state in self.optimizer.state.values()
                                     for value in state.values() if isinstance(value, torch.Tensor)]
        saved = [value.detach().clone() for value in tensors]
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._update(inputs, phase)
            current.wait_stream(stream)
            current.synchronize()
            with torch.no_grad():
                for value, original in zip(tensors, saved):
                    value.copy_(original)
            self.optimizer.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(self.device), torch.cuda.graph(graph):
                loss = self._update(inputs, phase)
            gradients = [parameter.grad for parameter in self.parameters]
        finally:
            torch.cuda.synchronize(self.device)
            with torch.no_grad():
                for value, original in zip(tensors, saved):
                    value.copy_(original)
        # Keep graph-owned scalar storage, not the autograd graph. Holding its
        # AccumulateGrad nodes across phase captures can bind stale CUDA streams.
        self.cache[phase] = (graph, inputs, loss.detach(), gradients)

    def step(self, batch, phase=None, transfer_done=None):
        if phase not in self.cache:
            self._capture(batch, phase)
        graph, inputs, loss, gradients = self.cache[phase]
        for key, value in inputs.items():
            source = batch[key].flatten(0, 1)
            if source.shape != value.shape or source.dtype != value.dtype:
                raise ValueError('CUDA learner replay requires fixed batch shapes and dtypes')
            value.copy_(source, non_blocking=True)
        if transfer_done is not None:
            transfer_done(self.device)
        for parameter, gradient in zip(self.parameters, gradients):
            parameter.grad = gradient
        graph.replay()
        return loss.detach().clone()
