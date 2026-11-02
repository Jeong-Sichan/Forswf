"""Narrow compatibility fix for the pinned KoRMo masking call.

The KoRMo remote module calls Transformers' create_causal_mask with
`inputs_embeds`, while Transformers 4.57.0 names that parameter `input_embeds`.
It also omits `cache_position`, which the remote module already used to make
`position_ids`. Recover the 1D cache position from those IDs.
Patch the already imported KoRMo module in memory; never edit cached model code.
"""

import importlib
import inspect


def fix_causal_mask_keyword(model) -> bool:
    if getattr(model.config, "model_type", None) != "kormo":
        return False
    module = importlib.import_module(type(model).__module__)
    function = getattr(module, "create_causal_mask", None)
    if function is None:
        return False
    parameters = inspect.signature(function).parameters
    if "input_embeds" not in parameters or "inputs_embeds" in parameters:
        return False

    def compatible_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            if "input_embeds" in kwargs:
                raise TypeError("both input_embeds and inputs_embeds were provided")
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        if "cache_position" in parameters and "cache_position" not in kwargs:
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                raise TypeError("KoRMo causal mask needs position_ids to recover cache_position")
            kwargs["cache_position"] = position_ids[0] if position_ids.ndim == 2 else position_ids
        return function(*args, **kwargs)

    module.create_causal_mask = compatible_mask
    return True
