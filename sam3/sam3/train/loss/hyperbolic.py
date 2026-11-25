import math

import torch


def _domain_eps(tensor: torch.Tensor, min_eps: float = 1e-5) -> float:
    if not torch.is_floating_point(tensor):
        return min_eps
    return max(min_eps, torch.finfo(tensor.dtype).eps * 4)


def exp_map0(x: torch.Tensor, curv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map Euclidean tangent vectors at the origin to Lorentz space.

    The returned tensor stores only the spatial coordinates; the time coordinate
    is reconstructed by the helpers below.
    """
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        curv = curv.float().clamp_min(eps)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        sqrt_curv = curv.sqrt()
        scaled_norm = sqrt_curv * x_norm
        sinh_input = torch.clamp(scaled_norm, min=eps, max=math.asinh(2**15))
        scale = torch.sinh(sinh_input) / torch.clamp(scaled_norm, min=eps)
        return scale * x


def time_component(x: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        curv = curv.float()
        return torch.sqrt(1.0 / curv + torch.sum(x * x, dim=-1, keepdim=True))


def pairwise_inner(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: torch.Tensor,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        y = y.float()
        curv = curv.float()
        x_time = time_component(x, curv)
        y_time = time_component(y, curv)
        return x @ y.T - x_time @ y_time.T


def pairwise_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        inner = pairwise_inner(x, y, curv)
        curv = curv.float()
        margin = _domain_eps(inner)
        acosh_arg = torch.clamp(-curv * inner, min=1.0 + margin)
        return torch.acosh(acosh_arg) / curv.sqrt()


def half_aperture(
    parent: torch.Tensor,
    curv: torch.Tensor,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=parent.device.type, enabled=False):
        parent = parent.float()
        curv = curv.float()
        margin = _domain_eps(parent)
        parent_norm = torch.norm(parent, dim=-1)
        asin_arg = 2.0 * min_radius / (parent_norm * curv.sqrt() + margin)
        return torch.asin(torch.clamp(asin_arg, min=-1.0 + margin, max=1.0 - margin))


def cone_angle(
    parent: torch.Tensor,
    child: torch.Tensor,
    curv: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exterior angle at parent for checking whether child is in its cone."""
    with torch.amp.autocast(device_type=parent.device.type, enabled=False):
        parent = parent.float()
        child = child.float()
        curv = curv.float()
        margin = _domain_eps(parent)
        parent_time = time_component(parent, curv).squeeze(-1)
        child_time = time_component(child, curv).squeeze(-1)
        lorentz_inner = torch.sum(parent * child, dim=-1) - parent_time * child_time
        curv_inner = curv * lorentz_inner

        numer = child_time + curv_inner * parent_time
        denom = torch.norm(parent, dim=-1) * torch.sqrt(
            torch.clamp(curv_inner * curv_inner - 1.0, min=margin)
        )
        acos_arg = numer / (denom + margin)
        return torch.acos(torch.clamp(acos_arg, min=-1.0 + margin, max=1.0 - margin))


def relative_cone_angle(
    parent: torch.Tensor,
    child: torch.Tensor,
    curv: torch.Tensor,
    aperture_scale: float = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the cone angle as a fraction of the parent's effective aperture."""
    angle = cone_angle(parent, child, curv, eps=eps)
    aperture = aperture_scale * half_aperture(
        parent,
        curv,
        min_radius=min_radius,
        eps=eps,
    )
    return angle / aperture.clamp_min(_domain_eps(aperture))


def signed_entailment_cone_margin(
    parent: torch.Tensor,
    child: torch.Tensor,
    curv: torch.Tensor,
    aperture_scale: float = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    angle = cone_angle(parent, child, curv, eps=eps)
    aperture = half_aperture(parent, curv, min_radius=min_radius, eps=eps)
    return aperture_scale * aperture - angle


def entailment_cone_loss(
    parent: torch.Tensor,
    child: torch.Tensor,
    curv: torch.Tensor,
    aperture_scale: float = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    margin = signed_entailment_cone_margin(
        parent,
        child,
        curv,
        aperture_scale=aperture_scale,
        min_radius=min_radius,
        eps=eps,
    )
    return torch.clamp(-margin, min=0.0)
