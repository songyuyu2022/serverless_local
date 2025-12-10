# instances_config.py
import json
import re

def load_instances(instances_path="instance/instances.json"):
    with open(instances_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 返回 {instance_id: instance_obj}
    return {inst["id"]: inst for inst in data}

def load_func_map(func_map_path="instance/func_map.json"):
    with open(func_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_experts_by_eid(instances_path, func_map_path, func_prefix="moe.expert_fwd:"):
    instances = load_instances(instances_path)
    func_map = load_func_map(func_map_path)

    groups = {}  # {eid: [instance_objs]}
    pattern = re.compile(rf"^{re.escape(func_prefix)}(\d+)$")

    for func_name, inst_ids in func_map.items():
        m = pattern.match(func_name)
        if not m:
            continue
        eid = int(m.group(1))
        groups.setdefault(eid, [])
        for iid in inst_ids:
            if iid in instances:
                groups[eid].append(instances[iid])

    return groups
