"""Piezas compartidas de la interfaz."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from motor.base import Mercado

TINTA, ROJO, VERDE, AMBAR, VIOLETA = "#2a62d6", "#d1344b", "#12a07a", "#b87a00", "#6250d6"
SERIES = [TINTA, ROJO, VERDE, AMBAR, VIOLETA]
TRAZOS = ["solid", "dash", "dot", "dashdot", "longdash"]
PAPEL, RENGLON, GRAFITO, GRIS = "#FBFCFE", "#DDE5F1", "#1F2838", "#5B6780"

CSS = """
<style>
/* El margen azul del cuaderno es el único gesto decorativo de la página */
[data-testid="stMain"] .block-container {
  border-left: 3px double #9DB4E0; padding-left: 2.2rem; max-width: 1180px; padding-top: 3.6rem;
}
[data-testid="stMain"] h1 { font-size: 2.05rem !important; color: #1B3FA0 !important; letter-spacing: -.01em; padding-bottom: .1rem !important; }
[data-testid="stMain"] h2 { font-size: 1.32rem !important; color: #B3202E !important; }
[data-testid="stMain"] h3 { font-size: 1.08rem !important; color: #1F2838 !important; }
[data-testid="stSidebar"] { border-right: 1px solid #D3DCEB; }
[data-testid="stSidebar"] h2 { color: #1B3FA0; font-size: 1.15rem !important; }
.lead { color: #5B6780; max-width: 74ch; margin: -.3rem 0 1rem; line-height: 1.5; }
.caja {
  border: 3px double #1B3FA0; padding: .55rem .9rem .6rem; background: #fff; margin-bottom: .75rem;
}
.caja .et { color: #5B6780; font-size: .86rem; line-height: 1.25; }
.caja .va { color: #1B3FA0; font-size: 1.62rem; font-weight: 700; line-height: 1.25;
            font-variant-numeric: tabular-nums; }
.caja .pi { color: #5B6780; font-size: .8rem; line-height: 1.3; }
.caja.sec { border: 1px solid #C6D2E6; }
.caja.sec .va { color: #1F2838; font-size: 1.3rem; }
.aviso { border-left: 3px solid #B3202E; padding: .55rem .85rem; background: #FCF1F2;
         margin: .4rem 0 .8rem; line-height: 1.45; }
.aviso.neutro { border-left-color: #1B3FA0; background: #EEF2F9; }
.aviso.modelo { border-left-color: #B87A00; background: #FBF5E8; }
.aviso b { display: block; margin-bottom: .1rem; }
[data-testid="stExpander"] summary p { font-weight: 600; }
@media (max-width: 640px) {
  [data-testid="stMain"] .block-container { padding-left: 1rem; padding-right: 1rem; border-left: none; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Entradas con memoria entre pestañas
# ---------------------------------------------------------------------------
def _ini(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def num(label, key, default, cont=None, **kw):
    _ini(key, default)
    kw.setdefault("format", "%.4f" if isinstance(default, float) else "%d")
    return (cont or st).number_input(label, key=key, **kw)


def sel(label, key, opciones, default=None, cont=None, **kw):
    _ini(key, default if default is not None else opciones[0])
    if st.session_state[key] not in opciones:
        st.session_state[key] = opciones[0]
    return (cont or st).selectbox(label, opciones, key=key, **kw)


def radio(label, key, opciones, default=None, cont=None, **kw):
    _ini(key, default if default is not None else opciones[0])
    if st.session_state[key] not in opciones:
        st.session_state[key] = opciones[0]
    kw.setdefault("horizontal", True)
    return (cont or st).radio(label, opciones, key=key, **kw)


def txt(label, key, default, cont=None, **kw):
    _ini(key, default)
    return (cont or st).text_input(label, key=key, **kw)


def casilla(label, key, default=False, cont=None, **kw):
    _ini(key, default)
    return (cont or st).checkbox(label, key=key, **kw)


def desliza(label, key, default, a, b, cont=None, **kw):
    _ini(key, default)
    return (cont or st).slider(label, a, b, key=key, **kw)


def lista_num(texto: str):
    """'0.25, 0.5' -> [0.25, 0.5]"""
    try:
        return [float(x) for x in str(texto).replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return None


TABLAS = "_tablas_editadas"


def tabla_editable(nombre, inicial, cont=None, **kw):
    """Tabla editable que conserva lo capturado al cambiar de pestaña."""
    ss = st.session_state
    base_k, ver_k, ult_k = f"_base_{nombre}", f"_ver_{nombre}", f"_ult_{nombre}"
    if base_k not in ss:
        ss[base_k] = inicial
        ss[ver_k] = 0
    ss.setdefault(TABLAS, set()).add(nombre)
    salida = (cont or st).data_editor(ss[base_k], key=f"ed_{nombre}_{ss[ver_k]}", **kw)
    ss[ult_k] = salida
    return salida


def reemplazar_tabla(nombre, nueva):
    ss = st.session_state
    ss[f"_base_{nombre}"] = nueva
    ss[f"_ver_{nombre}"] = ss.get(f"_ver_{nombre}", 0) + 1
    ss.pop(f"_ult_{nombre}", None)


def asentar_tablas():
    """Al cambiar de pestaña, lo capturado se vuelve la nueva base de la tabla."""
    ss = st.session_state
    for nombre in ss.get(TABLAS, set()):
        if f"_ult_{nombre}" in ss and nombre != "pagos":
            reemplazar_tabla(nombre, ss[f"_ult_{nombre}"])


# ---------------------------------------------------------------------------
# Mercado de la barra lateral
# ---------------------------------------------------------------------------
def mercado() -> Mercado:
    return st.session_state["mercado"]


def K_T():
    return st.session_state["K"], st.session_state["T"]


def encabezado(titulo, lead=""):
    st.title(titulo)
    if lead:
        st.markdown(f'<p class="lead">{lead}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
def f4(x, dec=4):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{dec}f}"


def caja(etiqueta, valor, pie="", secundaria=False, cont=None):
    v = valor if isinstance(valor, str) else f4(valor)
    pie_html = f'<div class="pi">{pie}</div>' if pie else ""
    (cont or st).markdown(f'<div class="caja{" sec" if secundaria else ""}"><div class="et">{etiqueta}</div>'
                          f'<div class="va">{v}</div>{pie_html}</div>', unsafe_allow_html=True)


def tabla(df, dec=4):
    """Tabla numérica con celdas vacías donde el dato no aplica."""
    salida = df.copy().astype(object)
    config = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            entero = pd.api.types.is_integer_dtype(df[c])
            salida[c] = df[c].map(lambda x: "" if pd.isna(x) else (f"{x:,d}" if entero else f"{x:,.{dec}f}"))
            config[c] = st.column_config.TextColumn(alignment="right")
    st.dataframe(salida, hide_index=True, width="stretch", column_config=config)


def aviso(titulo, texto="", clase="", cont=None):
    (cont or st).markdown(f'<div class="aviso {clase}"><b>{titulo}</b>{texto}</div>', unsafe_allow_html=True)


def seguro(fn, *a, **k):
    """Muestra errores de captura como mensaje y no como traza."""
    try:
        return fn(*a, **k)
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        aviso("No pude calcularlo", str(e))
    return None


def mostrar_estrategia(e):
    if not e.existe:
        aviso(e.titulo, e.nota, "neutro")
        return
    if e.tipo == "pura":
        aviso("Hay oportunidad de arbitraje", e.titulo)
    else:
        aviso("Diferencia contra el modelo, no es arbitraje", e.titulo, "modelo")
    c = st.columns(3)
    etq = "Ganancia hoy" if e.tipo == "pura" else "Diferencia contra el modelo"
    caja(etq, e.ganancia_hoy, cont=c[0])
    caja("Equivalente en T", e.ganancia_T, "capitalizada a la tasa libre de riesgo", True, cont=c[1])
    caja("Desviación del precio teórico", e.desviacion, secundaria=True, cont=c[2])
    st.markdown("**Estrategia**")
    tabla(pd.DataFrame(e.piernas))
    if e.nota:
        st.caption(e.nota)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def figura(titulo="", xt="", yt="", alto=360):
    f = go.Figure()
    f.update_layout(
        template="plotly_white", height=alto, title=dict(text=titulo, font=dict(size=15, color=GRAFITO), x=0, xanchor="left"),
        margin=dict(l=8, r=12, t=44 if titulo else 12, b=8), paper_bgcolor=PAPEL, plot_bgcolor=PAPEL,
        font=dict(family="Figtree, sans-serif", color=GRAFITO, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified")
    f.update_xaxes(title=xt, gridcolor=RENGLON, zerolinecolor="#B4C2DA", linecolor="#B4C2DA")
    f.update_yaxes(title=yt, gridcolor=RENGLON, zerolinecolor="#B4C2DA", linecolor="#B4C2DA", tickformat=",~r")
    return f


def lineas(f, x, series: dict):
    for i, (nom, y) in enumerate(series.items()):
        f.add_trace(go.Scatter(x=x, y=y, name=nom, mode="lines",
                               line=dict(color=SERIES[i % 5], width=2, dash=TRAZOS[i % 5])))
    f.update_layout(showlegend=len(series) > 1)
    return f


def marca_v(f, x, texto, color=GRIS):
    f.add_vline(x=x, line=dict(color=color, width=1, dash="dot"), annotation_text=texto,
                annotation_position="top", annotation_font=dict(size=11, color=GRIS))


def graficar(f):
    st.plotly_chart(f, width="stretch", config={"displaylogo": False, "displayModeBar": False, "scrollZoom": False})
