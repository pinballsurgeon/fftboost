from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from typing import Any


def sh(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.check_call(args)


def install_fftboost(
    prefer: str, version: str | None, ref: str | None, repo: str
) -> dict[str, Any]:
    # Always use the current interpreter (e.g., Jupyter kernel) for pip ops
    py = sys.executable
    # Clean existing install
    subprocess.call(
        [py, "-m", "pip", "uninstall", "-y", "fftboost"], stdout=subprocess.DEVNULL
    )

    if version:
        sh([py, "-m", "pip", "install", "-U", f"fftboost=={version}"])
        src = {"source": "pypi", "specifier": version}
    elif ref:
        sh(
            [
                py,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                f"git+{repo}@{ref}",
            ]
        )
        src = {"source": "git", "repo": repo, "ref": ref}
    else:
        if prefer == "git":
            sh(
                [
                    py,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    f"git+{repo}",
                ]
            )
            src = {"source": "git", "repo": repo, "ref": "HEAD"}
        else:
            sh([py, "-m", "pip", "install", "-U", "fftboost"])
            src = {"source": "pypi", "specifier": "latest"}
    return src


def verify_install(
    expect_evs: bool = True, expect_temporal_gating: bool = True
) -> dict[str, Any]:
    import fftboost

    path = getattr(fftboost, "__file__", "?")
    version = getattr(fftboost, "__version__", "?")

    # Optional code-signature checks of expected features
    try:
        import fftboost.automl as aml
        import fftboost.booster as bst

        evs_ok = "explained_variance" in inspect.getsource(aml)
        gating_ok = "m >= 1" in inspect.getsource(bst.Booster.fit)
    except Exception:
        evs_ok = False
        gating_ok = False

    info = {
        "path": str(path),
        "version": str(version),
        "evs_weighting_present": evs_ok,
        "temporal_gating_present": gating_ok,
    }

    # Print human-friendly summary
    print(json.dumps(info, indent=2))

    # Soft assertions (do not raise by default; return status instead)
    status = True
    if expect_evs and not evs_ok:
        status = False
    if expect_temporal_gating and not gating_ok:
        status = False
    info["ok"] = status
    return info


def print_notebook_snippet(version: str | None, ref: str | None, repo: str) -> None:
    if version:
        src = f"fftboost=={version}"
        installer = f"{src}"
    elif ref:
        installer = f"git+{repo}@{ref}"
    else:
        installer = "fftboost"
    code = (
        "import sys, subprocess, inspect\n\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall',"
        " '-y', 'fftboost'])  # clean\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install',"
        " '--no-cache-dir', '--force-reinstall', \""
        f"{installer}"
        '"])\n\n'
        "import fftboost\n"
        "print('fftboost:', getattr(fftboost, '__version__', '?'),"
        " getattr(fftboost, '__file__', '?'))\n"
        "import fftboost.automl as aml, fftboost.booster as bst\n"
        "print('EVS weight present:', 'explained_variance' in inspect.getsource(aml))\n"
        "print('Temporal experts gated after stage0:',"
        " 'm >= 1' in inspect.getsource(bst.Booster.fit))\n"
    )
    print(code.strip())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deterministic fftboost installer + verifier"
    )
    p.add_argument(
        "--prefer",
        choices=["pypi", "git"],
        default="pypi",
        help="Default source when neither --version nor --ref provided",
    )
    p.add_argument(
        "--version", default=None, help="Exact PyPI version to install (e.g., 1.1.0)"
    )
    p.add_argument(
        "--ref", default=None, help="Git ref to install (branch, tag, or commit SHA)"
    )
    p.add_argument(
        "--repo",
        default="https://github.com/pinballsurgeon/fftboost.git",
        help="Repository URL for git installs",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-install feature verification",
    )
    p.add_argument(
        "--print-notebook",
        action="store_true",
        help="Print a ready-to-paste notebook cell for this install",
    )
    args = p.parse_args()

    src = install_fftboost(args.prefer, args.version, args.ref, args.repo)
    print("Install source:", json.dumps(src))

    if args.print_notebook:
        print_notebook_snippet(args.version, args.ref, args.repo)

    if not args.no_verify:
        verify_install(expect_evs=True, expect_temporal_gating=True)


if __name__ == "__main__":
    main()
