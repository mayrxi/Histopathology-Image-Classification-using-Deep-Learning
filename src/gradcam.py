import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        loss = logits[:, class_idx]
        loss = loss.mean()
        loss.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2,3), keepdim=True)  # GAP on gradients
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # Normalize to [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-12)
        return cam, logits