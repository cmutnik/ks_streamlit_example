# Copyright (c) 2025 takotime808
import ast
import io
import re
import textwrap
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Optional SymPy for LaTeX parsing
try:
    from sympy.parsing.latex import parse_latex  # type: ignore
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

st.set_page_config(page_title="Markdown Equation → Python Function", page_icon="🧮", layout="centered")
st.title("🧮 Markdown/LaTeX Equation → Python Function")
st.caption("Type an equation like `$y = x^2 + 3x + 1$` or `$$f(x)=\\sin(x)+x^2$$`. I’ll render it and generate a Python function.")

# ------------------ Helpers ------------------
MATH_BLOCKS = [
    (r"\$\$(.+?)\$\$", re.DOTALL),
    (r"\$(.+?)\$", 0),
    (r"\\\((.+?)\\\)", 0),
    (r"\\\[(.+?)\\\]", re.DOTALL),
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
LATEX_CONSTS = {r"\pi": "pi", r"\mathrm{e}": "e", r"\E": "e"}
PY_BUILTINS_FUNCS = {"sin", "cos", "tan", "log", "exp", "sqrt"}
PY_CONSTS = {"pi", "e"}
IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")

def extract_first_math(md: str) -> Optional[str]:
    if not md:
        return None
    for pat, flags in MATH_BLOCKS:
        m = re.search(pat, md, flags)
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    for ln in lines:
        if any(ch in ln for ch in "=^_\\"):
            return ln
    return None

def strip_latex_wrappers(s: str) -> str:
    return re.sub(r"\\left|\\right", "", s)

def replace_frac_once(s: str) -> Tuple[str, bool]:
    pat = r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
    def repl(m):
        num, den = m.group(1), m.group(2)
        return f"(({num})/({den}))"
    new = re.sub(pat, repl, s, count=1)
    return new, (new != s)

def latex_to_python(expr: str) -> str:
    expr = strip_latex_wrappers(expr)
    changed = True
    while changed:
        expr, changed = replace_frac_once(expr)
    expr = expr.replace("^", "**")
    expr = re.sub(r"([A-Za-z])_\{([A-Za-z0-9]+)\}", r"\1\2", expr)
    expr = re.sub(r"([A-Za-z])_([A-Za-z0-9]+)", r"\1\2", expr)
    expr = expr.replace(r"\cdot", "*").replace(r"\times", "*")
    for k, v in LATEX_FUNCS.items():
        expr = re.sub(re.escape(k) + r"\s*\(", v + "(", expr)
    for k, v in LATEX_FUNCS.items():
        expr = re.sub(re.escape(k) + r"\s+([A-Za-z0-9_]+)", v + r"(\1)", expr)
    for k, v in LATEX_CONSTS.items():
        expr = expr.replace(k, v)
    expr = re.sub(r"\{([A-Za-z0-9_\.]+)\}", r"\1", expr)
    expr = re.sub(r"\\", "", expr)
    return re.sub(r"\s+", " ", expr).strip()

def split_equation(eq: str) -> Tuple[str, str]:
    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        return lhs.strip(), rhs.strip()
    return "y", eq.strip()

def infer_function_name(lhs: str) -> Tuple[str, List[str]]:
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z0-9_,\s]*)\s*\)\s*$", lhs)
    if m:
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        return name, args
    name = re.sub(r"[^A-Za-z0-9_]", "", lhs) or "f"
    return name, []

def infer_args(rhs_py: str, declared_args: List[str], lhs_name: str) -> List[str]:
    tokens = set(IDENT_RE.findall(rhs_py))
    tokens -= PY_BUILTINS_FUNCS
    tokens -= PY_CONSTS
    tokens.discard(lhs_name)
    tokens -= set(declared_args)
    tokens.discard("math")
    inferred = sorted(tokens)
    return declared_args + [t for t in inferred if t not in declared_args] if declared_args else inferred

def build_function_source(fn_name: str, args: List[str], rhs_py: str, doc_md: str) -> str:
    args_src = ", ".join(args) if args else ""
    body = f"return {rhs_py}" if rhs_py else "pass"
    doc = textwrap.indent((doc_md or "Auto-generated from Markdown/LaTeX.").strip(), " " * 4)
    return f"""def {fn_name}({args_src}):
    \"\"\"
{doc}
    \"\"\"
    {body}
"""

def validate_syntax(src: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(src)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def safe_exec_function(function_src: str, fn_name: str):
    safe_globals = {
        "__builtins__": {"abs": abs, "min": min, "max": max, "round": round},
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "log": np.log, "exp": np.exp, "sqrt": np.sqrt,
        "pi": np.pi, "e": np.e,
    }
    local_ns = {}
    exec(function_src, safe_globals, local_ns)
    return local_ns[fn_name]

def parse_user_value(s: str):
    s = s.strip()
    try:
        return float(s)
    except Exception:
        return eval(s, {"__builtins__": {}}, {"np": np, "pi": np.pi, "e": np.e})

def plot_1d(f, arg_name: str, x0: float):
    xs = np.linspace(x0 - 5, x0 + 5, 400)
    ys = np.array([f(x) for x in xs])
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(arg_name)
    plt.ylabel(f"f({arg_name})")
    plt.title(f"1D slice around {arg_name}={x0:.4g}")
    st.pyplot(fig)

def plot_2d(f, arg_names: List[str], x0: float, y0: float):
    xname, yname = arg_names[:2]
    X = np.linspace(x0 - 5, x0 + 5, 200)
    Y = np.linspace(y0 - 5, y0 + 5, 200)
    XX, YY = np.meshgrid(X, Y)
    Z = np.zeros_like(XX, dtype=float)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            Z[i, j] = f(XX[i, j], YY[i, j])
    fig = plt.figure()
    cs = plt.contourf(XX, YY, Z, levels=20)
    plt.xlabel(xname); plt.ylabel(yname)
    plt.title(f"2D contour around ({xname}={x0:.4g}, {yname}={y0:.4g})")
    plt.colorbar(cs)
    st.pyplot(fig)

# ------------------ Session init ------------------
for k, v in {
    "converted": False,
    "function_src": "",
    "fn_name": "",
    "args": [],
    "eq_block": "",
    "rhs_py": "",
}.items():
    st.session_state.setdefault(k, v)

# ------------------ Inputs ------------------
example = r"""Examples:
- $y = x^2 + 3x + 1$
- $$f(x) = \frac{\sin(x)}{1+x^2}$$
- \( V(r) = 4/3 \cdot \pi \cdot r^3 \)
- $g(t) = \exp(-t)\,\cos(2\pi t)$
"""
md = st.text_area("Markdown with an equation", height=160, placeholder=example, key="md_input")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("**Rendered equation**")
    current_eq = extract_first_math(st.session_state.get("md_input") or "")
    if current_eq:
        st.latex(current_eq)
    else:
        st.info("Enter an equation in $...$ or $$...$$ to render it here.")
with col2:
    st.markdown("**Conversion settings**")
    prefer_sympy = st.checkbox("Prefer SymPy LaTeX parser (if installed)", value=True)
    fn_name_override = st.text_input("Override function name (optional)", value="")

st.markdown("---")

# ------------------ Convert button ------------------
if st.button("Convert to Python function", type="primary", use_container_width=True, key="convert_btn"):
    eq_block = extract_first_math(st.session_state.get("md_input") or "")
    if not eq_block:
        st.error("No equation found. Put it inside $...$ or $$...$$.")
    else:
        lhs_raw, rhs_raw = split_equation(eq_block)
        rhs_py = None
        lhs_name, lhs_declared_args = infer_function_name(lhs_raw)

        if (prefer_sympy and SYMPY_AVAILABLE):
            try:
                sym = parse_latex(eq_block)
                s = str(sym)
                if s.startswith("Eq(") and "," in s:
                    inner = s[3:-1]
                    rhs_py = inner.split(",", 1)[1].strip()
                else:
                    rhs_py = str(sym)
            except Exception:
                rhs_py = None

        if not rhs_py:
            rhs_py = latex_to_python(rhs_raw)

        args = infer_args(rhs_py, lhs_declared_args, lhs_name)
        fn_name = re.sub(r"\W+", "", fn_name_override.strip()) if fn_name_override.strip() else (lhs_name or "f")
        function_src = build_function_source(fn_name, args, rhs_py, doc_md=eq_block)
        ok, err = validate_syntax(function_src)

        if not ok:
            st.error("❌ Generated code has a syntax error.")
            if err:
                st.code(err)
        else:
            # Persist to state so subsequent reruns keep results visible
            st.session_state.converted = True
            st.session_state.function_src = function_src
            st.session_state.fn_name = fn_name
            st.session_state.args = args
            st.session_state.eq_block = eq_block
            st.session_state.rhs_py = rhs_py

# ------------------ Render conversion results from state ------------------
if st.session_state.converted:
    st.subheader("Generated function")
    st.code(st.session_state.function_src, language="python")
    st.success("✅ Syntax looks good.")
    buf = io.BytesIO(st.session_state.function_src.encode("utf-8"))
    st.download_button("⬇️ Download .py", data=buf, file_name=f"{st.session_state.fn_name}.py", mime="text/x-python")

    # --- Sample runner & plots ---
    args = st.session_state.args
    fn_name = st.session_state.fn_name
    function_src = st.session_state.function_src

    if args:
        st.markdown("---")
        st.subheader("Try it and see plots")

        # Use a form; keep values in session via unique keys
        with st.form("sample_form", clear_on_submit=False):
            cols = st.columns(min(4, max(1, len(args))))
            input_vals = {}
            for i, a in enumerate(args):
                default_val = st.session_state.get(f"val_{a}", "1.0")
                with cols[i % len(cols)]:
                    input_vals[a] = st.text_input(f"{a} =", value=default_val, key=f"val_{a}")
            slice_vars = None
            if len(args) >= 3:
                st.caption("Pick two variables to plot a 2D contour slice (others fixed at the sample values).")
                c1, c2 = st.columns(2)
                sv1 = c1.selectbox("X-axis variable", args, index=0, key="sv1")
                sv2 = c2.selectbox("Y-axis variable", [a for a in args if a != sv1], index=0, key="sv2")
                slice_vars = (sv1, sv2)

            submitted = st.form_submit_button("Run sample")

        if submitted:
            try:
                f = safe_exec_function(function_src, fn_name)
                num_vals = [parse_user_value(st.session_state[f"val_{a}"]) for a in args]
                result = f(*num_vals)
                st.success(f"{fn_name}({', '.join(f'{a}={v}' for a, v in zip(args, num_vals))}) = {result}")

                if len(args) == 1:
                    plot_1d(f, args[0], float(num_vals[0]))
                elif len(args) == 2:
                    plot_2d(f, args, float(num_vals[0]), float(num_vals[1]))
                else:
                    if slice_vars:
                        xname, yname = slice_vars
                        x0 = float(num_vals[args.index(xname)])
                        y0 = float(num_vals[args.index(yname)])

                        def f2(x, y):
                            vals = num_vals.copy()
                            vals[args.index(xname)] = x
                            vals[args.index(yname)] = y
                            return f(*vals)

                        plot_2d(f2, [xname, yname], x0, y0)
                    else:
                        st.info("Choose two variables to draw a 2D contour slice.")
            except Exception as e:
                st.error(f"Runtime error: {e}")

st.markdown("---")
with st.expander("Notes & tips"):
    st.markdown(
        """
- Results persist using `st.session_state`, so submitting the form won’t “reset” the page.
- Put your equation in `$...$` or `$$...$$` so it can be detected and rendered.
- SymPy (if available) is preferred for parsing; otherwise a heuristic converter is used.
- Plots:
  - 1 variable → line slice around your sample.
  - 2 variables → 2D contour.
  - 3+ variables → pick two variables to slice; others fixed to your sample values.
"""
    )
