import numpy as np
import plotly.graph_objects as go
import streamlit as st

from motor import vanilla as va
from paginas.comun import K_T, caja, encabezado, figura, graficar, lineas, marca_v, mercado, radio, sel


@st.cache_data(show_spinner=False, max_entries=32)
def _griegas(m, K, T, tipo, estilo):
    if estilo == "europea":
        return va.griegas(lambda mm, tt: float(va.europea(mm, K, tt, tipo)), m, T)
    return va.griegas(lambda mm, tt: va.precio_estilo(mm, K, tt, tipo, "americana", 300), m, T)


@st.cache_data(show_spinner=False, max_entries=32)
def _curvas(m, K, T, tipo):
    xs = np.linspace(.5 * min(m.S0, K), 1.5 * max(m.S0, K), 61)
    filas = {k: [] for k in ("Delta", "Gamma", "Vega", "Theta", "Rho")}
    for x in xs:
        g = va.griegas_analiticas(m.con(S0=float(x)), K, T, tipo)
        for k in filas:
            filas[k].append(g[k])
    return xs, filas


@st.cache_data(show_spinner=False, max_entries=32)
def _superficie(m, K, T, tipo, griega):
    xs = np.linspace(.6 * min(m.S0, K), 1.4 * max(m.S0, K), 41)
    ts = np.linspace(max(T / 20, .01), T, 20)
    Z = np.array([[va.griegas_analiticas(m.con(S0=float(x)), K, float(t), tipo)[griega] for x in xs] for t in ts])
    return xs, ts, Z


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Griegas", "Complemento: no viene en las sesiones. Delta es la α de la cobertura cuando δ es pequeña.")
    c = st.columns(2)
    tipo = radio("Tipo", "w_g_t", ["Call", "Put"], cont=c[0]).lower()
    estilo = radio("Estilo", "w_g_e", ["Europea", "Americana"], cont=c[1]).lower()
    g = _griegas(m, K, T, tipo, estilo)
    c = st.columns(4)
    caja("Precio", g["Precio"], cont=c[0])
    caja("Delta", g["Delta"], "cambio por $1 en S", cont=c[1])
    caja("Gamma", g["Gamma"], "cambio de delta por $1 en S", cont=c[2])
    caja("Vega", g["Vega"] / 100, "por 1 punto de volatilidad", cont=c[3])
    c = st.columns(4)
    caja("Theta", g["Theta"] / 365, "por día calendario", True, cont=c[0])
    caja("Rho", g["Rho"] / 100, "por 1 punto de tasa", True, cont=c[1])
    caja("Psi", g["Psi"] / 100, "por 1 punto de rendimiento q", True, cont=c[2])
    caja("Elasticidad", g["Delta"] * m.S0 / g["Precio"] if g["Precio"] else 0, "cambio % por 1 % en S", True, cont=c[3])

    st.subheader("Perfil por nivel del subyacente")
    st.caption("Las gráficas usan las fórmulas europeas.")
    griega = sel("Griega", "w_g_g", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
    xs, cur = _curvas(m, K, T, tipo)
    fig = figura(f"{griega} contra el subyacente", "Subyacente", griega, 320)
    lineas(fig, xs, {griega: cur[griega]})
    marca_v(fig, m.S0, "S₀")
    marca_v(fig, K, "K")
    graficar(fig)

    x, t, Z = _superficie(m, K, T, tipo, griega)
    fig = figura(f"{griega} por subyacente y plazo", "Subyacente", "Plazo (años)", 380)
    escala = "RdBu_r" if griega in ("Delta", "Theta", "Rho") and Z.min() < 0 < Z.max() else [[0, "#EAF0FB"], [1, "#1B3FA0"]]
    medio = 0 if escala == "RdBu_r" else None
    fig.add_trace(go.Heatmap(x=x, y=t, z=Z, colorscale=escala, zmid=medio, colorbar=dict(title=griega, thickness=12),
                             hovertemplate="S = %{x:,.2f}<br>T = %{y:.2f}<br>%{z:,.4f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    graficar(fig)
