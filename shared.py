import msgpack, numpy as np, torch
def tensor_to_pack(t):
    a=t.detach().cpu().numpy()
    return {'shape':a.shape,'dtype':str(a.dtype),'data':a.tobytes()}
def pack_to_tensor(p, device="cpu"):
    a = np.frombuffer(p["data"], dtype=np.dtype(p["dtype"])).reshape(p["shape"])
    # 复制一份可写的
    a = np.array(a, copy=True)
    return torch.from_numpy(a).to(device)
def dumps(x): return msgpack.packb(x,use_bin_type=True)
def loads(b): return msgpack.unpackb(b,raw=False)
def route_pack(i,g): 
    return {'indices': tensor_to_pack(i), 'gates': tensor_to_pack(g)}
def route_unpack(p,device):
    return (pack_to_tensor(p['indices'],device), pack_to_tensor(p['gates'],device))
