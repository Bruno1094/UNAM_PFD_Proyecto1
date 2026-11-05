import numpy as np
import pandas as pd
import streamlit as st

from motor import curso as cu
from motor import vanilla as va
from paginas.comun import K_T, aviso, caja, encabezado, figura, graficar, lineas, marca_v, mercado, num, radio, seguro, sel, txt

PAGOS = {"S(T)ⁿ": ("S^n", True), "Call cuadrada": ("max(S-K,0)^2", False), "Put cuadrada": ("max(K-S,0)^2", False),
         "Digital call": ("M0*(S>K)", False), "Digital put": ("M0*(S<K)", False), "Digital en activo, call": ("S*(S>K)", False),
         "Digital en activo, put": ("S*(S<K)", False), "Fórmula propia": (None, False)}


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Límite y Black-Scholes", "La δ cuando se aproxima a 0, el árbol multiplicativo lleva a la dinámica lognormal y a Black-Scholes.")
    t1, t2, t3 = st.tabs(["Límite del árbol", "Black-Scholes", "Otros pagos X(S(T))"])

    with t1:
        st.latex(r"S_u=S_ne^{\mu\delta+\sigma\sqrt\delta},\quad S_d=S_ne^{\mu\delta-\sigma\sqrt\delta},\quad q=\frac{e^{r\delta}-d}{u-d}\doteq\frac12\left(1-\sqrt\delta\,\frac{\mu-r+\frac12\sigma^2}{\sigma}\right)+O(\delta)")
        c = st.columns(3)
        el = radio("Elección de μ", "w_l_e", ["Curso: μ = r - ½σ²", "Retorno esperado propio"], cont=c[0])
        mu_lim = m.b - .5 * m.sigma ** 2 if el.startswith("Curso") else num("μ", "w_l_mu", .10, c[1], step=.01)
        tp = radio("Tipo", "w_l_t", ["Call", "Put"], cont=c[2]).lower()
        ns = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500])
        vals = [va.vanilla_arbol(m, K, T, tp, "europea", int(x), metodo="notas", mu=mu_lim).precio for x in ns]
        qs = [va.factores(m, T, int(x), "notas", mu_lim)[2] for x in ns]
        apr = [.5 * (1 - np.sqrt(T / x) * (mu_lim - m.b + .5 * m.sigma ** 2) / m.sigma) for x in ns]
        bs = float(va.europea(m, K, T, tp))
        fig = figura("El precio del árbol tiende a Black-Scholes", "Periodos N (escala log)", "Precio", 330)
        lineas(fig, ns, {"Árbol": vals})
        fig.add_hline(y=bs, line=dict(color="#d1344b", dash="dash", width=2), annotation_text="Black-Scholes", annotation_font=dict(size=12))
        fig.update_xaxes(type="log", dtick=1)
        graficar(fig)
        fig = figura("q exacta y su aproximación para δ pequeña", "Periodos N (escala log)", "q", 280)
        lineas(fig, ns, {"q exacta": qs, "Aproximación": apr})
        fig.update_xaxes(type="log", dtick=1)
        graficar(fig)
        st.latex(r"S(t)=S(0)\exp\left[(r-\tfrac12\sigma^2)t+\sigma\sqrt tZ\right],\quad Z\sim N(0,1)")
        st.caption("Bajo Q, la distribución de S(t) depende de r y σ, pero no de μ. Por eso los modelos con distinta μ y la misma σ dan el mismo precio.")

    with t2:
        d = va.detalle_bs(m, K, T)
        c_, p_ = float(va.europea(m, K, T, "call")), float(va.europea(m, K, T, "put"))
        c = st.columns(3)
        caja("Call c(0)", c_, cont=c[0])
        caja("Put p(0)", p_, cont=c[1])
        caja("Paridad c - p", c_ - p_, f"debe dar f = {float(m.valor_forward(K, T)):,.4f}", True, cont=c[2])
        st.latex(r"c[S(0),T,K]=S(0)N(d_1)-Ke^{-rT}N(d_2),\qquad p[S(0),T,K]=Ke^{-rT}N(-d_2)-S(0)N(-d_1)")
        st.latex(r"d_1=\frac1{\sigma\sqrt T}\left[\ln\frac{S(0)}K+(r+\tfrac12\sigma^2)T\right],\qquad d_2=d_1-\sigma\sqrt T")
        st.latex(rf"d_1={d['d1']:.6f},\quad d_2={d['d2']:.6f},\quad N(d_1)={d['Nd1']:.6f},\quad N(d_2)={d['Nd2']:.6f}")
        if m.q or m.pagos:
            st.caption(f"Con ingresos, S(0) se reemplaza por el forward prepagado: {d['P']:,.4f}.")
        st.subheader("Lema de la sesión 7")
        st.latex(r"\int_b^\infty e^{ay}\frac1{\sqrt{2\pi}s}e^{-\frac{(y-m)^2}{2s^2}}dy=e^{am+\frac12a^2s^2}N(d),\qquad d=\frac{-b+m+as^2}{s}")
        st.caption("Con a = 1 sale el término de S(0) y con a = 0 el de K, usando b = ln(K/S(0)).")

    with t3:
        st.latex(r"X(0)=e^{-rT}\int_{-\infty}^\infty X(S(0)e^y)\frac1{\sigma\sqrt{2\pi T}}\exp\left(-\frac{(y-[r-\frac{\sigma^2}2]T)^2}{2\sigma^2T}\right)dy")
        c = st.columns(4)
        pago = sel("Pago", "w_x_p", list(PAGOS), cont=c[0])
        expr, usa_n = PAGOS[pago]
        ctes = {"K": K}
        if usa_n:
            ctes["n"] = num("n", "w_x_n", 2.0, c[1], step=1.0)
        if pago in ("Digital call", "Digital put"):
            ctes["M0"] = num("Pago M", "w_x_M", 100.0, c[1], min_value=0.0)
        if expr is None:
            expr = txt("X(S)", "w_x_e", "max(S-K,0)^2", cont=c[1])
        if m.pagos or m.q:
            aviso("Esta fórmula es sin dividendos", "Para esta sección elige Sin ingresos en la barra lateral.", "modelo")
        else:
            X = seguro(cu.valuar_lognormal, m.S0, m.rc, m.sigma, T, expr, **ctes)
            if X is not None:
                caja(f"Valor de {pago}", X, "integral numérica de la fórmula (5)")
                if usa_n:
                    n = ctes["n"]
                    st.latex(r"S^n(t)\exp\left(\left[\tfrac12\sigma^2n(n-1)+r(n-1)\right](T-t)\right)=" +
                             f"{m.S0 ** n * np.exp((.5 * m.sigma ** 2 * n * (n - 1) + m.rc * (n - 1)) * T):,.4f}")
