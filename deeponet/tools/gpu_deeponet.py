"""PyTorch implementation of the verified unstacked architecture and TF1 Adam rule."""
import torch


def strict_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Select DeepONet_GPU and set CUDA_VISIBLE_DEVICES=0.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda:0")


class UnstackedDeepONet(torch.nn.Module):
    def __init__(self, branch, trunk, activation="tanh", use_bias=True):
        super().__init__()
        if branch[-1] != trunk[-1]:
            raise ValueError("Branch and trunk feature widths must match")
        self.branch_sizes, self.trunk_sizes = tuple(branch), tuple(trunk)
        self.use_bias = use_bias
        self.activation = {"tanh":torch.tanh, "relu":torch.relu}[activation]
        self.branch_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(a,b)) for a,b in zip(branch[:-1],branch[1:])])
        self.branch_biases = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1,b), requires_grad=(use_bias or i<len(branch)-2))
            for i,b in enumerate(branch[1:])])
        self.trunk_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(a,b)) for a,b in zip(trunk[:-1],trunk[1:])])
        self.trunk_biases = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1,b)) for b in trunk[1:]])
        self.output_bias = torch.nn.Parameter(torch.zeros(()), requires_grad=use_bias)

    def tf_parameters(self):
        return (list(self.branch_weights)+list(self.branch_biases)+
                list(self.trunk_weights)+list(self.trunk_biases)+[self.output_bias])

    @torch.no_grad()
    def load_tf_arrays(self, arrays, prefix="step0_"):
        for i,parameter in enumerate(self.tf_parameters()):
            array = arrays[prefix+"param_{:02d}".format(i)]
            if tuple(array.shape) != tuple(parameter.shape):
                raise ValueError("TF/PyTorch parameter shape mismatch")
            parameter.copy_(torch.as_tensor(array, device=parameter.device))

    def encode_branch(self, inputs):
        value = inputs
        for i,(w,b) in enumerate(zip(self.branch_weights,self.branch_biases)):
            value = value@w
            if self.use_bias or i<len(self.branch_weights)-1:
                value = value+b
            if i<len(self.branch_weights)-1:
                value = self.activation(value)
        return value

    def encode_trunk(self, inputs):
        value = inputs
        for w,b in zip(self.trunk_weights,self.trunk_biases):
            value = self.activation(value@w+b)
        return value

    def forward(self, branch, trunk):
        output=(self.encode_branch(branch)*self.encode_trunk(trunk)).sum(dim=1,keepdim=True)
        return output+self.output_bias if self.use_bias else output


def normalized_mse(prediction, reference):
    # Keep the original ratio and float32 reduction; do not add an epsilon.
    return (prediction-reference).square().mean()/reference.square().mean()


class TF1Adam(torch.optim.Optimizer):
    """TF1 epsilon placement and float32 beta-power recurrence, with resumable slots.

    m <- m + (g-m)*(1-beta1); v <- v + (g*g-v)*(1-beta2)
    theta <- theta - lr*sqrt(1-beta2_power)/(1-beta1_power)*m/(sqrt(v)+eps)
    """
    def __init__(self, parameters, lr=.001, beta1=.9, beta2=.999, epsilon=1e-8):
        params = list(parameters)
        super().__init__(params, dict(lr=lr,beta1=beta1,beta2=beta2,epsilon=epsilon))
        device = params[0].device
        self.b1 = torch.tensor(beta1,device=device,dtype=torch.float32)
        self.b2 = torch.tensor(beta2,device=device,dtype=torch.float32)
        self.b1_power, self.b2_power = self.b1.clone(), self.b2.clone()
        self.updates = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Closure is not used by this TF1 Adam implementation")
        for group in self.param_groups:
            rate = group["lr"]*torch.sqrt(1-self.b2_power)/(1-self.b1_power)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                m,v = state["m"],state["v"]
                m.add_((gradient-m)*(1-self.b1))
                v.add_((gradient.square()-v)*(1-self.b2))
                parameter.sub_(rate*m/(torch.sqrt(v)+group["epsilon"]))
        self.b1_power.mul_(self.b1)
        self.b2_power.mul_(self.b2)
        self.updates += 1

    def state_dict(self):
        return dict(optimizer=super().state_dict(), updates=self.updates,
                    beta1_power=self.b1_power.detach().cpu(),
                    beta2_power=self.b2_power.detach().cpu())

    def load_state_dict(self, state):
        super().load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.b1_power.copy_(state["beta1_power"].to(self.b1_power.device))
        self.b2_power.copy_(state["beta2_power"].to(self.b2_power.device))
