def select_loss(losses, key, *, use_default=False):
    if use_default:
        assert "default" in losses, "Default loss requested but not configured"
        return losses["default"]

    if key in losses:
        return losses[key]

    assert key != "all", "Loss must be specified for key='all'"
    assert "default" in losses, (
        f"Key {key} not found in losses, and no default provided"
    )
    return losses["default"]
