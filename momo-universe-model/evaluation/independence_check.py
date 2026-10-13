"""Static import-graph check (SPEC 5.5): inference/ and model/ must not
import anything under data/ (teacher code). Pure-assert script.
"""
import ast
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def imports_of(path):
    tree = ast.parse(open(path).read())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(("." * node.level) + (node.module or ""))
    return mods


def check():
    bad = []
    for pkg in ("inference", "model"):
        for fn in os.listdir(os.path.join(ROOT, pkg)):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(ROOT, pkg, fn)
            for m in imports_of(p):
                top = m.lstrip(".").split(".")[0]
                assert top != "data", f"{p} imports forbidden module '{m}'"
                if pkg == "inference":
                    # inference may only use model/ + stdlib + torch/numpy
                    assert top in ("model", "torch", "numpy", "json", "os",
                                   "dataclasses", "typing", ""), \
                        f"{p} imports unexpected module '{m}'"
    return True


if __name__ == "__main__":
    check()
    print("PASS independence_check: inference/ and model/ import no data/ modules")
