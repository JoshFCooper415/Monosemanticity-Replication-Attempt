import torch
import torch.nn.functional as F

class ActivationCollector(torch.nn.Module):
    def __init__(self, model, layer_num, eps=1e-12):
        super().__init__()
        self.model = model
        self.layer_num = layer_num
        self.activations = None
        self.eps = eps  # Small value to avoid division by zero

    def forward(self, input_ids, attention_mask=None):
        def hook(module, input, output):
            # Detach and move to CPU to avoid unnecessary GPU memory usage
            act = output[0].detach().cpu()
            
            # Apply L2 normalization across the feature dimension
            norm = torch.norm(act, p=2, dim=-1, keepdim=True)
            self.activations = act / (norm + self.eps)

        layer = self.model.model.layers[self.layer_num]
        handle = layer.register_forward_hook(hook)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        handle.remove()
        return outputs, self.activations

    def get_normalized_activations(self):
        if self.activations is None:
            raise ValueError("No activations collected yet. Run a forward pass first.")
        return self.activations