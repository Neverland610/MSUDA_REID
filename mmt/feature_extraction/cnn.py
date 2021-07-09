from __future__ import absolute_import
from collections import OrderedDict

import torch
from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None, domain_label=-1, epochs=-1):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()
    if modules is None:
        if domain_label == -1:
            if epochs == -1:
                outputs = model(inputs)
            else:
                outputs = model(inputs, epochs=epochs)
        else:
            if epochs == -1:
                outputs = model(inputs, domain_label * torch.ones(inputs.shape[0], dtype=torch.long).cuda(),)
            else:
                outputs = model(inputs, domain_label * torch.ones(inputs.shape[0], dtype=torch.long).cuda(),
                                epochs=epochs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    if domain_label == -1:
        if epochs == -1:
            model(inputs)
        else:
            model(inputs, epochs=epochs)
    else:
        if epochs == -1:
            model(inputs, domain_label * torch.ones(inputs.shape[0], dtype=torch.long).cuda())
        else:
            model(inputs, domain_label * torch.ones(inputs.shape[0], dtype=torch.long).cuda(),
                  epochs=epochs)
    for h in handles:
        h.remove()
    return list(outputs.values())

