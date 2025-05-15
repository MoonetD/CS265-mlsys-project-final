import torchvision.models as models, torch.fx as fx
from collections import Counter

model = models.resnet152()          # training mode by default
gm     = fx.symbolic_trace(model)

# crude classifier: anything that isn't a placeholder/get_attr/output
# and is created before sep_backward is a forward activation
activations = [
    n for n in gm.graph.nodes
    if n.op not in ('placeholder', 'get_attr', 'output')
]

print("total FX nodes:", len(gm.graph.nodes))
print("activation-ish nodes:", len(activations))
