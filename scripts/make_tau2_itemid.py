#!/usr/bin/env python
"""Generate the AgentItemId file for a tau2 training or evaluation run.

AgentItemId files are generated rather than committed (the repo's .gitignore
excludes the directory), so this script is the definition of the mapping.

The index in ``item_id`` is positional, and the environment server resolves it
with the same rule -- domains in the order given, tasks in the loader's order,
one running counter across all of them:

    item_id "tau2_7"  ->  self._task_ids[7]  ->  (domain, task_id)

Which means the domain list and split here must match ``TAU2_DOMAIN`` and
``TAU2_TASK_SPLIT`` on the server, or every row trains on a different task than
it names. Run it inside the tau2 environment (Python >=3.12):

    python scripts/make_tau2_itemid.py --domains retail airline telecom \
        --split train --out data/

Omitting --out prints to stdout.
"""
import argparse
import json
import sys
from pathlib import Path


def build(domains, split):
    from tau2.registry import registry

    rows = []
    for domain in domains:
        tasks = registry.get_tasks_loader(domain)(split)
        if not tasks:
            sys.exit(f"tau2: no tasks for domain '{domain}' split '{split}'")
        for task in tasks:
            rows.append({
                "item_id": f"tau2_{len(rows)}",
                "task_type": domain,
                "task_id": str(task.id),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", required=True,
                    help="domain names, in the same order as TAU2_DOMAIN")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", type=Path,
                    help="directory to write tau2_<domains>_<split>.json into")
    args = ap.parse_args()

    rows = build(args.domains, args.split)
    text = json.dumps(rows, indent=2)

    if args.out is None:
        print(text)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"tau2_{'-'.join(args.domains)}_{args.split}.json"
    path.write_text(text + "\n")
    print(f"{len(rows)} tasks -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
