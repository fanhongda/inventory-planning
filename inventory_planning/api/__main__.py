"""
Run the intake API.

    python -m inventory_planning.api --port 8000

Binds to localhost by default and stays there. This is one planner's machine until an
authentication layer exists in front of it, and every declaration it writes is
attributed to whatever `by` the caller supplied — an audit field with nobody checking
it. Binding to 0.0.0.0 without that check would make the attribution decorative.

That got more serious when the macro scalars became editable. A caller who reaches this
port can change `cycle_stock_basis` or `safety_stock_exposure`, which restate every
figure the next run produces, and sign the change with any name. The change is refused
unless it loads, and it is recorded with a reason — but recorded against a name nobody
verified. The identity seam in INTERFACE.md §7 is what closes this; until it does, the
bind address is the whole of the access control.
"""

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inventory_planning.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="Config directory")
    parser.add_argument("--store", default=None,
                        help="Store root. Overrides INVENTORY_PLANNING_STORE — the "
                             "store path is the isolation between a dev run and a real "
                             "one, exactly as a branch is for code.")
    parser.add_argument("--output", default=None,
                        help="Output directory. The run registry lives under it, so "
                             "this is where the policy screen reads runs from. "
                             "Defaults to the workspace's, which is what makes "
                             "--tenant reach it.")
    parser.add_argument("--tenant", default=None,
                        help="Which workspace to serve. --config / --store / --output "
                             "still win where given.")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "Serving needs uvicorn. Install the optional extra:\n"
            "    pip install -e '.[api]'")

    from .app import create_app

    uvicorn.run(create_app(config_dir=args.config, store_root=args.store,
                           output_dir=args.output, tenant=args.tenant),
                host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
