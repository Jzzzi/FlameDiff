import torch

def process_ckp(ckp_path, out_path):
    ckp = torch.load(ckp_path)
    state_dict = ckp['state_dict']
    new_state_dict = {k.replace("_model.", ""): v for k, v in state_dict.items()}
    torch.save(new_state_dict, out_path)

process_ckp('/data15/jinkun.liu.2502/CodeSpace/FlameDiff/weights/EnDecoder.ckpt', '/data15/jinkun.liu.2502/CodeSpace/FlameDiff/weights/EnDecoder_new.ckpt')