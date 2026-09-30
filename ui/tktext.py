"""
ui/tktext.py
============
Text that every Tk installation can draw.

Tk on a lab server often has only the X11 core fonts (helvetica, courier,
fixed ...), which cover Latin-1 and nothing else: "Å" and "°" are drawn,
but "→", "σ", "Γ", "−" or the superscript minus of "Å⁻¹" come out as a
literal ``\\u2192``. The labels in this program -- and the axis labels read
from the files -- use those characters freely, so rather than rewrite
every string, :func:`install` makes the Tk widgets translate them to plain
ASCII on the way in (``Å⁻¹`` -> ``Å^-1``, ``σ`` -> ``sigma``, ``→`` ->
``->`` ...). The data, the saved files and the matplotlib plots (which
bring their own fonts) keep the real characters.

Set the environment variable ``ARPES_UNICODE=1`` to switch this off on a
machine whose Tk does have Unicode fonts.
"""
import os

#: Longest first where one is a prefix of another.
REPLACEMENTS = [
    ("⁻¹", "^-1"), ("⁻²", "^-2"), ("⁻³", "^-3"),
    ("⁻", "^-"), ("⁰", "^0"), ("⁴", "^4"),
    ("→", "->"), ("←", "<-"), ("↔", "<->"), ("⇒", "=>"),
    ("↑", "up"), ("↓", "down"),
    ("−", "-"), ("–", "-"), ("—", "--"), ("…", "..."),
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ("•", "*"), ("∘", "o"), ("≈", "~"), ("≠", "!="),
    ("≤", "<="), ("≥", ">="), ("∞", "inf"), ("√", "sqrt"),
    ("∂", "d"), ("∇", "grad"), ("∥", "par"), ("⊥", "perp"),
    ("⚠", "(!)"), ("ħ", "hbar"), ("ₑ", "e"), ("₀", "0"),
    ("σ", "sigma"), ("Σ", "Sigma"), ("Γ", "Gamma"),
    ("Δ", "Delta"), ("δ", "delta"), ("ε", "eps"),
    ("α", "alpha"), ("β", "beta"), ("γ", "gamma"),
    ("θ", "theta"), ("φ", "phi"), ("ϕ", "phi"), ("η", "eta"),
    ("χ", "chi"), ("μ", "u"), ("ν", "nu"), ("π", "pi"),
    ("λ", "lambda"), ("ω", "omega"), ("Ω", "Omega"),
    ("τ", "tau"), ("ρ", "rho"), ("κ", "kappa"),
]


def ascii_safe(text):
    """``text`` with everything outside Latin-1 spelt in ASCII."""
    if not isinstance(text, str):
        return text
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        pass
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
    return "".join(ch if ord(ch) < 256 else "?" for ch in text)


_KEYS = ("text", "label", "title", "message", "detail")
_installed = []


def _clean(value):
    if isinstance(value, str):
        return ascii_safe(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_clean(v) for v in value)
    return value


def install():
    """Patch tkinter so every text reaching Tk is drawable. Idempotent."""
    if _installed or os.environ.get("ARPES_UNICODE") == "1":
        return
    import tkinter
    from tkinter import ttk

    original_options = tkinter.Misc._options

    def _options(self, cnf, kw=None):
        def fix(d):
            if not d:
                return d
            d = dict(d)
            for key in _KEYS:
                if key in d:
                    d[key] = _clean(d[key])
            if "values" in d and not isinstance(self, ttk.Combobox):
                d["values"] = _clean(d["values"])
            return d
        return original_options(self, fix(cnf) if isinstance(cnf, dict) else cnf,
                                fix(kw) if isinstance(kw, dict) else kw)
    tkinter.Misc._options = _options

    original_format = ttk._format_optdict

    def _format_optdict(optdict, script=False, ignore=None):
        optdict = dict(optdict)
        for key in _KEYS + ("values",):
            if key in optdict:
                optdict[key] = _clean(optdict[key])
        return original_format(optdict, script, ignore)
    ttk._format_optdict = _format_optdict

    original_title = tkinter.Wm.wm_title

    def wm_title(self, string=None):
        return original_title(self, _clean(string))
    tkinter.Wm.wm_title = tkinter.Wm.title = wm_title

    original_text_insert = tkinter.Text.insert

    def text_insert(self, index, chars, *args):
        return original_text_insert(self, index, _clean(chars), *args)
    tkinter.Text.insert = text_insert

    original_list_insert = tkinter.Listbox.insert

    def list_insert(self, index, *elements):
        return original_list_insert(self, index, *[_clean(e) for e in elements])
    tkinter.Listbox.insert = list_insert

    original_menu_add = tkinter.Menu.add

    def menu_add(self, itemType, cnf={}, **kw):
        cnf = dict(cnf)
        cnf.update(kw)
        if "label" in cnf:
            cnf["label"] = _clean(cnf["label"])
        return original_menu_add(self, itemType, cnf)
    tkinter.Menu.add = menu_add

    # Combobox values keep their real text as far as the program is
    # concerned (ui.tkbase.Form maps between the two); what Tk shows is the
    # safe spelling.
    _installed.append(True)
