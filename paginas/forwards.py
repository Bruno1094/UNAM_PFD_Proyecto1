import numpy as np
import pandas as pd
import streamlit as st

from motor import vanilla as va
from paginas.comun import K_T, caja, encabezado, figura, graficar, lineas, marca_v, mercado, num


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Forwards y paridad put-call", "Calculadora de Fowards")
    t1, t2, t3, t4 = st.tabs(["Precio y valor", "Comparar ingresos", "Payoffs", "Paridad put-call"])

    with t1:
        B = float(m.B(T))
        F, f = float(m.forward(T)), float(m.valor_forward(K, T))
        c = st.columns(3)
        caja("Precio forward F", F, "el K que hace valer cero al contrato", cont=c[0])
        caja("Valor del forward largo f", f, f"con K = {K:,.4f}", cont=c[1])
        caja("Valor del forward corto", -f, secundaria=True, cont=c[2])
        with st.expander("Cómo se obtuvo", expanded=True):
            if m.pagos:
                st.markdown("Portafolio A: forward largo y efectivo KB(0,T). Portafolio B: una unidad de subyacente, menos C, el valor presente del ingreso.")
                st.latex(r"C=\sum_iD_iB(0,t_i)=" + f"{m.vp_ingresos(T):,.4f}")
                st.latex(rf"F=\frac{{S_0-C}}{{B(0,T)}}=\frac{{{m.S0:,.4f}-{m.vp_ingresos(T):,.4f}}}{{{B:.6f}}}={F:,.4f}")
                st.latex(rf"f=S_0-C-KB(0,T)={f:,.4f}")
            elif m.q:
                st.markdown("Con dividendos, A unidades de hoy se vuelven $B_E^{-1}(0,T)A$ unidades en T, así que basta tener $B_E(0,T)$ unidades hoy.")
                st.latex(rf"F=S_0B_E(0,T)B^{{-1}}(0,T)={m.S0:,.4f}\times{float(m.BE(T)):.6f}\div{B:.6f}={F:,.4f}")
                st.latex(rf"f=S_0B_E(0,T)-KB(0,T)={f:,.4f}")
            else:
                st.markdown("Portafolio A: forward largo y efectivo KB(0,T). Portafolio B: una unidad de subyacente. Ambos valen S(T) en T.")
                st.latex(rf"f+KB(0,T)=S_0\ \Rightarrow\ f=S_0-KB(0,T)={f:,.4f}")
                st.latex(rf"F=S_0B^{{-1}}(0,T)={m.S0:,.4f}\div{B:.6f}={F:,.4f}")
        ts = np.linspace(max(T / 60, .01), max(2 * T, .25), 60)
        fig = figura("Precio forward por plazo", "Plazo (años)", "F", 300)
        graficar(lineas(fig, ts, {"F(0,T)": [float(m.forward(x)) for x in ts]}))

    with t2:
        st.caption("El mismo contrato bajo cada tipo de ingreso.")
        c = st.columns(2)
        q_alt = num("Rendimiento de dividendos (%)", "w_fw_q", 2.0, c[0], step=.25, format="%.3f") / 100
        D = num("Pago fijo por trimestre", "w_fw_D", 1.0, c[1], min_value=0.0, step=.25)
        trim = tuple((x, D) for x in np.arange(.25, T + 1e-9, .25))
        base = m.con(q=0.0, pagos=())
        casos = {"Sin dividendos": base, "Con dividendos": base.con(q=q_alt), "Ingreso fijo": base.con(pagos=trim)}
        filas = [{"Caso": n, "S₀ ajustado": float(x.prepago(T)), "Precio forward F": float(x.forward(T)),
                  "Valor f con K": float(x.valor_forward(K, T))} for n, x in casos.items()]
        st.dataframe(pd.DataFrame(filas).style.format({a: "{:,.4f}" for a in list(filas[0])[1:]}), hide_index=True, width="stretch")

    with t3:
        xs = np.linspace(.5 * K, 1.5 * K, 121)
        fig = figura("Payoff de las posiciones largas", "S(T)", "Payoff")
        lineas(fig, xs, {"Forward": xs - K, "Call": np.maximum(xs - K, 0), "Put": np.maximum(K - xs, 0)})
        marca_v(fig, K, "K")
        graficar(fig)
        st.markdown("""
| | Largo | Corto |
|---|---|---|
| Forward | $S_T-K$, cobertura ante subidas | $K-S_T$, cobertura ante caídas |
| Call | $(S_T-K)_+$, derecho a comprar | $-(S_T-K)_+$, obligación de vender |
| Put | $(K-S_T)_+$, derecho a vender | $-(K-S_T)_+$, obligación de comprar |
""")
        st.caption("El payoff de un derivado es un juego de suma cero: lo que gana una parte lo pierde la otra.")

    with t4:
        st.markdown("Portafolio A: call largo y put corto. Portafolio B: forward largo. Ambos pagan $S_T-K$, así que hoy valen lo mismo.")
        c_, p_ = float(va.europea(m, K, T, "call")), float(va.europea(m, K, T, "put"))
        c = st.columns(3)
        caja("c - p con Black-Scholes", c_ - p_, secundaria=True, cont=c[0])
        caja("Forward f", float(m.valor_forward(K, T)), secundaria=True, cont=c[1])
        caja("Diferencia", c_ - p_ - float(m.valor_forward(K, T)), "debe ser 0", True, cont=c[2])
        st.latex(r"c(S_0,T,K)-p(S_0,T,K)=f(S_0,T,K)=S_0-B(0,T)K")
        st.latex(r"\text{Con dividendos: }c-p=S_0B_E(0,T)-KB(0,T)\qquad\text{Con ingreso fijo: }c-p=S_0-C-KB(0,T)")
        st.caption("Para comprobarla con precios de mercado, usa la pestaña Arbitraje.")
