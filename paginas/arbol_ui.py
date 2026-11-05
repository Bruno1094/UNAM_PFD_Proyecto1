"""Definición del árbol, compartida por Árboles, Martingalas, Americanas y Arbitraje."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from motor import curso as cu
from paginas.comun import aviso, caja, casilla, figura, graficar, lista_num, num, radio, sel, txt

MODOS = ["Multiplicativo", "Recombinante por fechas", "Nodo por nodo"]
PARAM = ["Dar u y d", "Con σ y r (μ = r - ½σ²)", "Con σ, r y μ"]
PAGOS = {"Call": "max(S-K,0)", "Put": "max(K-S,0)", "Forward": "S-K", "Digital call": "P*(S>K)", "Digital put": "P*(S<K)",
         "Digital en activo, call": "S*(S>K)", "Digital en activo, put": "S*(S<K)", "Call cuadrada": "max(S-K,0)^2",
         "Put cuadrada": "max(K-S,0)^2", "Asiática call": "max(A-K,0)", "Asiática put": "max(K-A,0)",
         "Lookback: máximo": "M", "Doble sin toque": "P*(m>B1)*(M<B2)", "Fórmula propia": None}
MAPA_MODO = {"mult": MODOS[0], "fechas": MODOS[1], "nodos": MODOS[2]}


def _cargar():
    s = st.session_state
    e = cu.EJEMPLOS[s["w_t_ej"]]
    s["w_t_modo"] = MAPA_MODO[e["modo"]]
    s["w_t_pago"] = e["pago"]
    s["w_t_K"] = e.get("K", 0.0)
    s["w_t_estilo"] = e["estilo"]
    s["w_t_w"] = e["w"]
    pares = {"tabla": "w_t_tabla", "precios": "w_t_nodos", "param": "w_t_param", "S0": "w_t_S0", "u": "w_t_u", "d": "w_t_d",
             "N": "w_t_N", "r": "w_t_r", "T": "w_t_T", "expr": "w_t_expr", "B1": "w_t_B1", "B2": "w_t_B2", "monto": "w_t_P"}
    for a, b in pares.items():
        if a in e:
            s[b] = e[a]
    if "B" in e:
        s["w_t_B"] = e["B"]
    if "sigma" in e:
        s["w_t_sig"] = e["sigma"]


def _construir():
    modo = radio("Tipo de árbol", "w_t_modo", MODOS)
    if modo == MODOS[0]:
        par = sel("Cómo se eligen u y d", "w_t_param", PARAM)
        c = st.columns(4)
        S0 = num("S(0)", "w_t_S0", 100.0, c[0], min_value=.0001)
        N = int(num("Periodos N", "w_t_N", 3, c[1], min_value=1, max_value=10))
        if par == PARAM[0]:
            u = num("u", "w_t_u", 1.2, c[2], min_value=.0001, step=.05)
            d = num("d", "w_t_d", .9, c[3], min_value=.0001, step=.05)
            B_txt = txt("B por periodo (uno solo, o uno por fecha separados por coma)", "w_t_B", "0.99")
            B = lista_num(B_txt)
            if not B:
                raise ValueError("Escribe B como número, por ejemplo 0.99.")
        else:
            T = num("Vencimiento T (años)", "w_t_T", 1.0, c[2], min_value=.001, step=.25)
            sig = num("Volatilidad σ", "w_t_sig", .20, c[3], min_value=.0001, step=.01)
            r = num("Tasa r", "w_t_r", .05, step=.01)
            if par == PARAM[2]:
                mu = num("Retorno esperado μ", "w_t_mu", .10, step=.01)
                u, d, Bv = cu.parametros("propia", r, sig, T / N, mu)
            else:
                u, d, Bv = cu.parametros("sigma_r", r, sig, T / N)
            st.latex(rf"\delta=\tfrac TN={T / N:.4f},\quad u=e^{{\mu\delta+\sigma\sqrt\delta}}={u:.6f},\quad d=e^{{\mu\delta-\sigma\sqrt\delta}}={d:.6f},\quad B(\,,\,)=e^{{-r\delta}}={Bv:.6f}")
            B = [Bv]
        return cu.multiplicativo(S0, u, d, N, B)
    B_txt = txt("B por periodo (uno solo, uno por fecha, o uno por nodo no final)", "w_t_B", "1")
    B = lista_num(B_txt)
    if not B:
        raise ValueError("Escribe B como número, por ejemplo 1.")
    if modo == MODOS[1]:
        t = st.text_area("Precios por fecha, de abajo hacia arriba. Una fecha por renglón.", key="w_t_tabla",
                         value=None if "w_t_tabla" in st.session_state else "100\n80, 120\n60, 100, 140\n40, 80, 120, 160", height=118)
        filas = [lista_num(x) for x in str(t).strip().splitlines() if x.strip()]
        if any(f is None for f in filas) or len(filas) < 2:
            raise ValueError("Escribe un renglón por fecha, con los precios separados por coma.")
        return cu.por_fechas(filas, B)
    t = txt("Precios S₁, S₂, S₃, ... en el orden de los nodos", "w_t_nodos", "80, 60, 120, 36, 72, 80, 180")
    st.caption("Del nodo j salen 2j (abajo) y 2j+1 (arriba). Con N = 2 son 7 precios; con N = 3, 15.")
    pr = lista_num(t)
    if not pr:
        raise ValueError("Escribe los precios separados por coma.")
    return cu.por_nodos(pr, B)


def _pago():
    c = st.columns(3)
    pago = sel("Pago X", "w_t_pago", list(PAGOS), cont=c[0])
    K = num("Strike K", "w_t_K", 100.0, c[1])
    ctes = {"K": K}
    if pago in ("Digital call", "Digital put"):
        ctes["P"] = num("Pago M", "w_t_P", 100.0, c[2])
    if pago == "Doble sin toque":
        ctes["P"] = num("Pago", "w_t_P", 100.0, c[2])
        cc = st.columns(2)
        ctes["B1"] = num("B₁", "w_t_B1", 50.0, cc[0])
        ctes["B2"] = num("B₂", "w_t_B2", 250.0, cc[1])
    expr = PAGOS[pago]
    if expr is None:
        expr = txt("Fórmula de X", "w_t_expr", "max(S-K,0)", cont=c[2])
        st.caption("Variables: S, S0, K, A (promedio desde S(0)), G (geométrico), M (máximo), m (mínimo), Sm (precio a la mitad). "
                   "Funciones: max, min, abs, exp, log, sqrt, si(condición, a, b).")
    return expr, ctes


def definir(titulo=None, forzar=None, inicial=None):
    """Dibuja los campos. Regresa dict(a, v, expr, ctes) o None si hay un error de captura."""
    s = st.session_state
    if "w_t_modo" not in s:
        s.setdefault("w_t_ej", inicial or next(iter(cu.EJEMPLOS)))
        _cargar()
    cont = st.expander(titulo, expanded=False) if titulo else st.container()
    with cont:
        c = st.columns([3, 1])
        c[0].selectbox("Ejemplo usado", list(cu.EJEMPLOS), key="w_t_ej")
        c[1].button("Cargar ejemplo", on_click=_cargar, width="stretch", key=f"cargar_{titulo}")
        try:
            a = _construir()
            expr, ctes = _pago()
            if forzar:
                estilo = forzar
            else:
                estilo = radio("Estilo", "w_t_estilo", ["Europea", "Americana"]).lower()
            v = cu.valuar(a, expr, estilo, **ctes)
        except ValueError as e:
            aviso("Revisa los datos del árbol", str(e))
            return None
    if v.problemas:
        lista = ", ".join(str(x) for x in v.problemas[:8])
        aviso("q queda fuera de (0, 1)", f"En los nodos {lista} no se cumple S_d < S_n B⁻¹ < S_u: ahí hay arbitraje y el resultado no es un precio válido.")
    return {"a": a, "v": v, "expr": expr, "ctes": ctes}


def dibujar(v, marcar=None, alto=None):
    """Árbol con número de nodo, S y V. Se dibuja completo."""
    a = v.a
    N = a.N
    y = np.zeros(2 ** (N + 1))
    finales = cu.nivel(N)
    y[finales] = np.arange(2 ** N) - (2 ** N - 1) / 2
    for k in range(N - 1, -1, -1):
        for j in cu.nivel(k):
            y[j] = .5 * (y[2 * j] + y[2 * j + 1])
    fig = figura("", "Fecha", "", alto or (170 + 44 * 2 ** N))
    ruta = set()
    if marcar:
        j = 1
        for c in marcar:
            nuevo = 2 * j + (c == "u")
            ruta.add((j, nuevo))
            j = nuevo
    for j in range(1, 2 ** N):
        k = cu.tiempo(j)
        for hijo in (2 * j, 2 * j + 1):
            fig.add_trace(go.Scatter(x=[k, k + 1], y=[y[j], y[hijo]], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color="#d1344b" if (j, hijo) in ruta else "#C6D2E6", width=2.5 if (j, hijo) in ruta else 1)))
    grupos = {"Continuar": ([], "#2a62d6", "circle"), "Ejercer": ([], "#d1344b", "diamond")}
    for j in range(1, 2 ** (N + 1)):
        grupos["Ejercer" if v.ejercer[j] else "Continuar"][0].append(j)
    for nom, (lista, color, simb) in grupos.items():
        if not lista:
            continue
        xs = [cu.tiempo(j) for j in lista]
        textos = [f"<b>{j}</b>  S = {a.S[j]:,.2f}<br>V = {v.V[j]:,.3f}" for j in lista]
        hover = []
        for j in lista:
            h = f"Nodo {j} ({cu.nombre_nodo(j)})<br>S = {a.S[j]:,.4f}<br>V = {v.V[j]:,.4f}"
            if j < 2 ** N:
                h += f"<br>q = {v.q[j]:.4f}<br>α = {v.alfa[j]:,.4f}<br>β = {v.beta[j]:,.4f}"
            hover.append(h)
        fig.add_trace(go.Scatter(x=xs, y=[y[j] for j in lista], mode="markers+text", name=nom, text=textos, textposition="top center",
                                 hovertext=hover, hoverinfo="text", textfont=dict(size=11),
                                 marker=dict(size=9, color=color, symbol=simb, line=dict(color="#fff", width=1.5))))
    rango = (2 ** N) / 2
    fig.update_yaxes(visible=False, range=[-rango, rango + .4])
    fig.update_xaxes(tickvals=list(range(N + 1)), ticktext=[f"{k}δ" for k in range(N + 1)], range=[-.4, N + .4], showgrid=False)
    fig.update_layout(hovermode="closest", showlegend=bool(v.ejercer.any()))
    return fig


def tabla_nodos(v):
    a = v.a
    filas = []
    for j in range(1, 2 ** (a.N + 1)):
        f = {"Nodo": j, "ω": cu.nombre_nodo(j), "Fecha": f"{cu.tiempo(j)}δ", "S": a.S[j], "V": v.V[j]}
        if j < 2 ** a.N:
            f.update({"q": v.q[j], "α": v.alfa[j], "β": v.beta[j]})
        filas.append(f)
    return pd.DataFrame(filas)
