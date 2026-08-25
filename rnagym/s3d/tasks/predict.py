"""Run model prediction adapters over sharded benchmark targets."""

import importlib
import sys

from rnagym.s3d.models.utils import Adapter


def _load_adapter(environment: str) -> Adapter:
    """Load the adapter installed in one model environment."""
    module = importlib.import_module(f"rnagym.s3d.models.{environment}")
    return module.ADAPTER


def main() -> None:
    """Set up, check, or run one model prediction shard."""
    environment, action = sys.argv[1:3]
    adapter = _load_adapter(environment)
    if action == "kinds":
        print("\n".join(adapter.kinds))
        return
    if action == "setup":
        if adapter.setup:
            adapter.setup()
        return

    kind = sys.argv[3]
    if kind not in adapter.kinds:
        raise ValueError(f"{environment} does not predict {kind}")
    targets = adapter.targets(kind)
    if action == "check":
        missing = sum(not adapter.complete(target) for target in targets)
        print(f"Missing {missing}/{len(targets)} {environment} {kind} predictions")
        raise SystemExit(not targets or bool(missing))

    shard, num_shards = map(int, sys.argv[4:6])
    assigned = targets[shard::num_shards]
    assigned = [target for target in assigned if not adapter.complete(target)]
    if not assigned:
        print("All assigned predictions are complete")
        return
    if action == "predict":
        adapter.predict(assigned, kind, shard)
    elif action == "prepare" and adapter.prepare:
        adapter.prepare(assigned, kind)
    else:
        raise ValueError(f"Unknown {environment} task: {action}")


if __name__ == "__main__":
    main()
