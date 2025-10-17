import torch, json
import torch.nn.functional as F

class PVTrialAbort(RuntimeError):
    """Raised inside a trial forward when any (layer, head) violates the L1 bound."""
    pass

def precision_metric(quant_o, fa2_o, verbose=True, round_num=4): 
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float() 
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}

def _parse_float_list(s: str):
    """Accepts comma-separated floats or a JSON list string; returns List[float] or None."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # try JSON first
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return [float(x) for x in arr]
        except Exception:
            pass
    # fallback: comma-separated
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [float(p) for p in parts] if parts else None
