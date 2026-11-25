"""Shared linear heads and L1 optimization."""
from __future__ import annotations
import numpy as np

# Shared linear heads for classification and discrete-time survival.
import torch
from torch import nn
from torch.nn import functional as F


class UnifiedLinearModel(nn.Module):
    """Task-specific likelihoods share the same linear predictor."""

    def __init__(self, n_features: int, n_outputs: int, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.linear = nn.Linear(n_features, n_outputs, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = self.linear.weight.device
            logits = self(torch.as_tensor(x, dtype=torch.float64, device=device))
            if self.task_type == "multiclass":
                probabilities = logits.softmax(dim=1)
            else:
                probabilities = logits.sigmoid()
                if self.task_type == "binary":
                    probabilities = torch.cat((1.0 - probabilities, probabilities), dim=1)
            return probabilities.cpu().numpy()

    def predict_survival(self, x: np.ndarray) -> np.ndarray:
        if self.task_type != "survival":
            raise ValueError("Survival curves require a survival model")
        return np.cumprod(1.0 - self.predict_proba(x), axis=1)


def survival_targets(times, events, cuts) -> dict[str, np.ndarray]:
    """Encode finite intervals; events after the last endpoint are censored there.

    ``cuts`` contains every interval's right endpoint, including the finite
    administrative horizon. Events exactly at an endpoint belong to that bin.
    Censoring within a bin contributes only fully observed earlier bins.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    cuts = np.asarray(cuts, dtype=float)
    if cuts.ndim != 1 or len(cuts) == 0 or not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("Survival endpoints must be finite and strictly increasing")
    bins = np.searchsorted(cuts, times, side="left")
    targets = np.zeros((len(times), len(cuts)), dtype=float)
    mask = np.zeros_like(targets)
    for i, (time, event, event_bin) in enumerate(zip(times, events, bins)):
        if event and event_bin < len(cuts):
            mask[i, :event_bin + 1] = 1.0
            targets[i, event_bin] = 1.0
        else:
            mask[i] = cuts <= time
    return {"targets": targets, "mask": mask}


def linear_data_loss(logits, y, task_type, class_weights=None):
    if task_type == "binary":
        weights = None if class_weights is None else class_weights[y.long()]
        return F.binary_cross_entropy_with_logits(logits[:, 0], y, weight=weights)
    if task_type == "multiclass":
        return F.cross_entropy(logits, y, weight=class_weights)
    if task_type == "survival":
        # Sum intervals per patient, then average patients, not observed bins.
        return (F.binary_cross_entropy_with_logits(logits, y["targets"], reduction="none")
                * y["mask"]).sum(dim=1).mean()
    raise ValueError(f"Unknown task type: {task_type}")


def fit_linear(x, y, *, task_type="binary", n_outputs=None, l1_strength=0.0,
               optimizer="proximal", lr=1.0, max_iter=2000, tol=1e-7,
               device="cpu", initial_state=None, class_weight="none"):
    """Fit all heads with one full-batch loop; intercepts are never penalized.

    Balanced classification weights are computed from these training cases only.
    Survival uses its unweighted censoring likelihood.

    Proximal convergence uses its gradient mapping. Adam stops on a sustained
    objective plateau; this is explicitly reported rather than claiming KKT
    convergence. Returned models reside on CPU for portable serialization.
    """
    if task_type not in {"binary", "multiclass", "survival"}:
        raise ValueError(f"Unknown task type: {task_type}")
    if optimizer not in {"adam", "proximal"}:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    if l1_strength < 0 or lr <= 0 or max_iter < 1:
        raise ValueError("Require nonnegative L1, positive learning rate and iterations")
    if class_weight not in {"none", "balanced"}:
        raise ValueError("class_weight must be none or balanced")
    if task_type == "survival" and class_weight != "none":
        raise ValueError("Class balancing is only defined for classification")
    x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)
    if task_type == "survival":
        y_tensor = {key: torch.as_tensor(y[key], dtype=torch.float64, device=device)
                    for key in ("targets", "mask")}
        inferred_outputs = y_tensor["targets"].shape[1]
    else:
        y_tensor = torch.as_tensor(y, dtype=torch.long if task_type == "multiclass"
                                   else torch.float64, device=device)
        inferred_outputs = int(y_tensor.max().item()) + 1 if task_type == "multiclass" else 1
    class_weights = None
    if class_weight == "balanced":
        n_classes = 2 if task_type == "binary" else (n_outputs or inferred_outputs)
        counts = torch.bincount(y_tensor.long(), minlength=n_classes)
        if torch.any(counts == 0):
            raise ValueError("Each class must occur in the training fold")
        # Mean case weight is one, preserving the scale of the L1 penalty.
        class_weights = len(y_tensor) / (n_classes * counts.to(torch.float64))
    model = UnifiedLinearModel(x_tensor.shape[1], n_outputs or inferred_outputs, task_type).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    adam = torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam" else None
    previous = float("inf")
    plateau = 0
    converged = False
    step = float(lr)
    residual = float("nan")
    for iteration in range(1, max_iter + 1):
        model.zero_grad(set_to_none=True)
        data_loss = linear_data_loss(model(x_tensor), y_tensor, task_type, class_weights)
        objective = data_loss + l1_strength * model.linear.weight.abs().sum()
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite training objective")
        if adam is not None:
            objective.backward()
            adam.step()
        else:
            data_loss.backward()
            old = [p.detach().clone() for p in model.parameters()]
            gradients = [p.grad.detach().clone() for p in model.parameters()]
            # Backtracking majorizes the smooth likelihood, with exact L1 prox.
            with torch.no_grad():
                for _ in range(60):
                    for index, (parameter, value, grad) in enumerate(zip(model.parameters(), old, gradients)):
                        update = value - step * grad
                        if index == 0:
                            update = update.sign() * (update.abs() - step * l1_strength).clamp_min(0)
                        parameter.copy_(update)
                    differences = [p - value for p, value in zip(model.parameters(), old)]
                    candidate = linear_data_loss(model(x_tensor), y_tensor, task_type, class_weights)
                    bound = data_loss.detach() + sum((g * d).sum() + d.square().sum() / (2 * step)
                                                      for g, d in zip(gradients, differences))
                    if candidate <= bound + 1e-12:
                        break
                    step *= 0.5
                else:
                    raise RuntimeError("Proximal backtracking did not find a descent step")
                residual = float(torch.sqrt(sum(d.square().sum() for d in differences)).item() / step)
        with torch.no_grad():
            current = float((linear_data_loss(model(x_tensor), y_tensor, task_type, class_weights)
                             + l1_strength * model.linear.weight.abs().sum()).item())
        if optimizer == "proximal":
            converged = residual <= tol
            step *= 1.2
        else:
            plateau = plateau + 1 if abs(previous - current) <= tol * max(1.0, abs(current)) else 0
            converged = plateau >= 20
        previous = current
        if converged:
            break
    model.cpu()
    return model, {"class_weight": class_weight,
                   "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
                   "optimizer": optimizer, "iterations": iteration, "converged": converged,
                   "objective": current, "gradient_mapping_norm": residual,
                   "stop_reason": ("gradient_mapping" if optimizer == "proximal" else "objective_plateau")
                                  if converged else "max_iter"}


def survival_on_times(hazards, cuts, times) -> np.ndarray:
    """Evaluate survival at shared times using completed finite intervals.

    Each hazard corresponds to one finite endpoint in ``cuts``. The survival
    curve is held constant beyond the administrative horizon; no extrapolated
    event probability is assigned beyond that horizon.
    """
    hazards = np.asarray(hazards, dtype=float)
    cuts = np.asarray(cuts, dtype=float)
    times = np.asarray(times, dtype=float)
    if hazards.ndim != 2 or hazards.shape[1] != len(cuts):
        raise ValueError("Expected one hazard per finite interval endpoint")
    survival = np.concatenate((np.ones((hazards.shape[0], 1)),
                               np.cumprod(1 - hazards, axis=1)), axis=1)
    return survival[:, np.searchsorted(cuts, times, side="right")]


