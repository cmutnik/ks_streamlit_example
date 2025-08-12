# Copyright (c) 2025 takotime808
import ast
import io
import re
import textwrap
from typing import List, Tuple, Optional

import streamlit as st

# Optional sympy (better LaTeX parsing if installed)
try:
    from sympy.parsing.latex import parse_latex  # type: ignore
    from sympy import symbols, lambdify, SympifyError # noqa
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

st.set_page_config(page_title="Markdown Equation → Python Function", page_icon="🧮", layout="centered")
st.title("🧮 Markdown/LaTeX Equation → Python Function")
st.caption("Type an equation like `$y = x^2 + 3x + 1$` or `$$f(x)=\\sin(x)+x^2$$`. I’ll render it and generate a Python function.")

# --- Helpers -----------------------------------------------------------------

MATH_BLOCKS = [
    (r"\$\$(.+?)\$\$", re.DOTALL),  # $$ ... $$
    (r"\$(.+?)\$", 0),              # $ ... $
    (r"\\\((.+?)\\\)", 0),          # \( ... \)
    (r"\\\[(.+?)\\\]", re.DOTALL),  # \[ ... \]
]

LATEX_FUNCS = {
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\log": "log",
    r"\ln": "log",
    r"\exp": "exp",
    r"\sqrt": "sqrt",
}

LATEX_CONSTS = {
    r"\pi": "pi",
    r"\mathrm{e}": "e",
    r"\E": "e",
}

def extract_first_math(md: str) -> Optional[str]:
    if not md:
        return None
    for pat, flags in MATH_BLOCKS:
        m = re.search(pat, md, flags)
        if m:
            return m.group(1).strip()
    # fallback: entire line if it looks like an equation
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    for ln in lines:
        if any(ch in ln for ch in "=^_\\"):
            return ln
    return None

def strip_latex_wrappers(s: str) -> str:
    # Remove \left \right and similar wrappers
    s = re.sub(r"\\left|\\right", "", s)
    return s

def replace_frac_once(s: str) -> Tuple[str, bool]:
    # Replace a single \frac{a}{b} occurrence with ((a)/(b))
    pattern = r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
    def repl(m):
        num, den = m.group(1), m.group(2)
        return f"(({num})/({den}))"
    new = re.sub(pattern, repl, s, count=1)
    return new, (new != s)

def latex_to_python(expr: str) -> str:
    expr = strip_latex_wrappers(expr)

    # Replace \frac recursively
    changed = True
    while changed:
        expr, changed = replace_frac_once(expr)

    # Replace ^ with **
    expr = expr.replace("^", "**")

    # Subscripts: x_{1} -> x1
    expr = re.sub(r"([A-Za-z])_\{([A-Za-z0-9]+)\}", r"\1\2", expr)
    expr = re.sub(r"([A-Za-z])_([A-Za-z0-9]+)", r"\1\2", expr)

    # Multiplication
    expr = expr.replace(r"\cdot", "*").replace(r"\times", "*")

    # Functions with parentheses
    for k, v in LATEX_FUNCS.items():
        expr = re.sub(re.escape(k) + r"\s*\(", v + "(", expr)

    # Functions without parentheses
    for k, v in LATEX_FUNCS.items():
        expr = re.sub(re.escape(k) + r"\s+([A-Za-z0-9_]+)", v + r"(\1)", expr)

    # Constants
    for k, v in LATEX_CONSTS.items():
        expr = expr.replace(k, v)

    # Remove braces around single tokens
    expr = re.sub(r"\{([A-Za-z0-9_\.]+)\}", r"\1", expr)

    # Remove stray backslashes
    expr = re.sub(r"\\", "", expr)

    return re.sub(r"\s+", " ", expr).strip()


def split_equation(eq: str) -> Tuple[str, str]:
    """Return (lhs, rhs). If no '=', treat as y = eq."""
    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        return lhs.strip(), rhs.strip()
    return "y", eq.strip()

def infer_function_name(lhs: str) -> Tuple[str, List[str]]:
    """
    Infer function name and explicit args from lhs like:
    y            -> ('y', [])
    f(x)         -> ('f', ['x'])
    g(x, y, z)   -> ('g', ['x','y','z'])
    """
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z0-9_,\s]*)\s*\)\s*$", lhs)
    if m:
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        return name, args
    # Otherwise, simple variable name
    name = re.sub(r"[^A-Za-z0-9_]", "", lhs) or "f"
    return name, []

PY_BUILTINS_FUNCS = {"sin","cos","tan","log","exp","sqrt"}
PY_CONSTS = {"pi","e"}

IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")

def infer_args(rhs_py: str, declared_args: List[str], lhs_name: str) -> List[str]:
    tokens = set(IDENT_RE.findall(rhs_py))
    tokens -= PY_BUILTINS_FUNCS
    tokens -= PY_CONSTS
    tokens.discard(lhs_name)
    tokens -= set(declared_args)
    # Likely math module names etc.
    tokens.discard("math")
    # Sort for stability, but keep declared args first
    inferred = sorted(tokens)
    if declared_args:
        return declared_args + [t for t in inferred if t not in declared_args]
    return inferred

def build_function_source(fn_name: str, args: List[str], rhs_py: str, doc_md: str) -> str:
    args_src = ", ".join(args) if args else ""
    body = f"return {rhs_py}" if rhs_py else "pass"
    doc = textwrap.indent(doc_md.strip() or "Auto-generated from Markdown/LaTeX.", " " * 4)
    func = f"""def {fn_name}({args_src}):
    \"\"\"
{doc}
    \"\"\"
    {body}
"""
    return func

def validate_syntax(src: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(src)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# --- UI ----------------------------------------------------------------------

example = r"""Example inputs:
- $y = x^2 + 3x + 1$
- $$f(x) = \frac{\sin(x)}{1+x^2}$$
- \( V(r) = 4/3 \cdot \pi \cdot r^3 \)
- $g(t) = \exp(-t)\,\cos(2\pi t)$
"""

md = st.text_area("Markdown with an equation", height=160, placeholder=example)

col1, col2 = st.columns([1,1])#, vertical_alignment="center")

with col1:
    st.markdown("**Rendered equation**")
    eq_block = extract_first_math(md or "")
    if eq_block:
        # Show nicely
        st.latex(eq_block)
    else:
        st.info("Enter an equation in $...$ or $$...$$ to render it here.")

with col2:
    st.markdown("**Conversion settings**")
    prefer_sympy = st.checkbox("Prefer SymPy LaTeX parser (if installed)", value=True)
    fn_name_override = st.text_input("Override function name (optional)", value="")

st.markdown("----")

if st.button("Convert to Python function", type="primary", use_container_width=True):
    if not eq_block:
        st.error("No equation found. Put it inside $...$ or $$...$$.")
    else:
        lhs_raw, rhs_raw = split_equation(eq_block)

        # Try SymPy first if available and requested
        rhs_py = None
        lhs_name, lhs_declared_args = infer_function_name(lhs_raw)

        if (prefer_sympy and SYMPY_AVAILABLE):
            try:
                sym = parse_latex(eq_block)
                # If parse_latex got the whole equation with Eq(lhs, rhs)
                sym_str = str(sym)
                # Detect something like Eq(lhs, rhs)
                if sym_str.startswith("Eq(") and "," in sym_str:
                    # crude split: Eq(lhs, rhs)
                    inner = sym_str[3:-1]
                    parts = inner.split(",", 1)
                    rhs_py = parts[1].strip()
                else:
                    # Maybe user provided only RHS expression
                    rhs_py = str(sym)
            except Exception:
                rhs_py = None

        # If SymPy failed or not used, fallback heuristic
        if not rhs_py:
            rhs_py = latex_to_python(rhs_raw)

        # Infer args
        args = infer_args(rhs_py, lhs_declared_args, lhs_name)
        if fn_name_override.strip():
            fn_name = re.sub(r"\W+", "", fn_name_override.strip())
        else:
            fn_name = lhs_name if lhs_name else "f"

        # Build source
        function_src = build_function_source(fn_name, args, rhs_py, doc_md=eq_block)

        ok, err = validate_syntax(function_src)

        st.subheader("Generated function")
        st.code(function_src, language="python")

        if ok:
            st.success("✅ Syntax looks good.")
            # Offer a quick test form if there are arguments
            if args:
                with st.expander("Try it with sample values"):
                    defaults = {}
                    test_inputs = {}
                    cols = st.columns(min(4, max(1, len(args))))
                    for i, a in enumerate(args):
                        with cols[i % len(cols)]:
                            test_inputs[a] = st.text_input(f"{a} =", value="1.0", key=f"arg_{a}")
                    # Build a callable in a safe namespace
                    local_ns = {}
                    try:
                        exec(function_src, {}, local_ns)
                        f = local_ns[fn_name]
                        # Convert inputs to float where possible
                        call_vals = []
                        for a in args:
                            v = test_inputs[a]
                            try:
                                call_vals.append(float(v))
                            except Exception:
                                call_vals.append(eval(v, {"pi": 3.141592653589793, "e": 2.718281828459045}))
                        if st.button("Run sample", key="run_sample"):
                            try:
                                res = f(*call_vals)
                                st.success(f"{fn_name}({', '.join(map(str, call_vals))}) = {res}")
                            except Exception as e:
                                st.error(f"Runtime error: {e}")
                    except Exception as e:
                        st.warning(f"Could not create a test runner: {e}")

            # Download
            buf = io.BytesIO(function_src.encode("utf-8"))
            st.download_button("⬇️ Download .py", data=buf, file_name=f"{fn_name}.py", mime="text/x-python")
        else:
            st.error("❌ Generated code has a syntax error.")
            if err:
                st.code(err)

st.markdown("----")
with st.expander("Notes & tips"):
    st.markdown(
        """
- Put your equation in `$ ... $` or `$$ ... $$` so it can be detected and rendered.
- Supported LaTeX features (heuristic fallback): `^`, `\\frac{...}{...}`, `\\sin`, `\\cos`, `\\tan`, `\\log`, `\\ln`, `\\exp`, `\\sqrt`, `\\cdot`, `\\times`, subscripts like `x_{1}`.
- If you have SymPy installed, check **Prefer SymPy** for more robust LaTeX parsing.
- The function name is inferred from the left-hand side (e.g., `f(x) = ...` ⇒ `def f(x): ...`). If it’s just `y = ...`, the name defaults to `y` with arguments inferred from the right-hand side.
"""
    )
