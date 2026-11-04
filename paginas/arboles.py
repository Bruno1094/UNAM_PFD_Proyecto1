import numpy as np
import pandas as pd
import streamlit as st

from motor import curso as cu
from motor import vanilla as va
from paginas import arbol_ui
from paginas.comun import aviso, caja, encabezado, graficar, num, seguro, tabla, txt


def pagina():
    encabezado("Árbol binomial", "Valuación por réplica: de un periodo a N periodos, con la numeración de nodos y la notación de clase.")
    t1, t2, t3, t4, t5 = st.tabs(["Un periodo", "N periodos", "Réplica en una trayectoria", "Fórmula binomial", "No recombinante"])

    with t1:
        st.caption("Sesión 2. Cualquier derivado es un vector (X_u, X_d).")
        c = st.columns(3)
        S0 = num("S₀", "w_b_S", 40.0, c[0], min_value=.001)
        Su = num("S_u", "w_b_Su", 42.0, c[1], min_value=.001)
        Sd = num("S_d", "w_b_Sd", 38.0, c[2], min_value=.001)
        c = st.columns(3)
        Xu = num("X_u", "w_b_Xu", 3.0, c[0])
        Xd = num("X_d", "w_b_Xd", 0.0, c[1])
        B = num("B(0,T)", "w_b_B", float(np.exp(-.08 / 12)), c[2], min_value=.0001, step=.01, format="%.6f")
        if Su <= Sd:
            aviso("Revisa los escenarios", "S_u debe ser mayor que S_d.")
        else:
            r = va.bernoulli(S0, Su, Sd, Xu, Xd, B)
            c = st.columns(4)
            caja("Valor V₀", r["V0"], cont=c[0])
            caja("q", r["q"], secundaria=True, cont=c[1])
            caja("α", r["alfa"], "venta en corto" if r["alfa"] < 0 else "compra de subyacente", True, cont=c[2])
            caja("β", r["beta"], "préstamo" if r["beta"] < 0 else "ahorro", True, cont=c[3])
            st.latex(r"\alpha S_u+B^{-1}(0,T)\beta=X_u,\qquad\alpha S_d+B^{-1}(0,T)\beta=X_d")
            st.latex(r"\alpha=\frac{X_u-X_d}{S_u-S_d},\quad\beta=B(0,T)[X_u-\alpha S_u],\quad q=\frac{S_0B^{-1}(0,T)-S_d}{S_u-S_d},\quad V_0=B(0,T)[qX_u+(1-q)X_d]")
            if not 0 < r["q"] < 1:
                aviso("q queda fuera de (0, 1)", "Hay arbitraje entre el subyacente y la cuenta en efectivo. La estrategia está en Arbitraje, sección Árbol.")
            else:
                st.caption("q no depende del derivado, solo del subyacente y de la cuenta en efectivo. Tampoco interviene la probabilidad real.")

    with t2:
        d = arbol_ui.definir()
        if d:
            v = d["v"]
            c = st.columns(4)
            caja("Valor en t = 0", v.precio, cont=c[0])
            caja("q₁", v.q[1], secundaria=True, cont=c[1])
            caja("α₁", v.alfa[1], secundaria=True, cont=c[2])
            caja("β₁", v.beta[1], secundaria=True, cont=c[3])
            st.latex(r"V_n=B(i\delta,(i+1)\delta)[q_nV_u+(1-q_n)V_d],\qquad q_n=\frac{S_nB^{-1}(i\delta,(i+1)\delta)-S_d}{S_u-S_d}")
            if d["a"].N <= 5:
                graficar(arbol_ui.dibujar(v))
                st.caption("En cada nodo: número de nodo, S y V. Pasa el cursor para ver q, α y β.")
            else:
                st.caption("Con más de 5 periodos el árbol no se dibuja; todos los nodos están en la tabla de abajo.")
            with st.expander("Todos los nodos", expanded=d["a"].N > 5):
                tabla(arbol_ui.tabla_nodos(v))

    with t3:
        st.caption("Sesión 3. Usa el árbol de la pestaña N periodos. Elige una trayectoria y comprueba que la cobertura se autofinancia y replica el pago.")
        if d:
            v = d["v"]
            N = d["a"].N
            w = txt(f"Trayectoria ω, con {N} letras u o d", "w_t_w", "uud").replace("0", "").strip().lower()
            if len(w) != N or set(w) - {"u", "d"}:
                aviso("Revisa la trayectoria", f"Escribe {N} letras, por ejemplo {'ud' * (N // 2)}{'u' * (N % 2)}.")
            else:
                filas = cu.seguir(v, w)
                if N <= 5:
                    graficar(arbol_ui.dibujar(v, marcar=w))
                tabla(pd.DataFrame(filas))
                ok = v.estilo == "europea" and all(abs(f["Cartera anterior vale"] - f["Cartera nueva cuesta"]) < 1e-8 for f in filas[1:-1])
                fin = abs(filas[-1]["Cartera anterior vale"] - filas[-1]["Valor del derivado"]) < 1e-8
                if ok and fin:
                    aviso("La cobertura funciona en esta trayectoria",
                          "En cada fecha la cartera anterior vale justo lo que cuesta la nueva, y al vencimiento iguala el pago. "
                          "No hizo falta poner ni sacar dinero.", "neutro")
                elif v.estilo == "americana":
                    st.caption("En la americana puede haber consumo: ve la pestaña Americanas.")

    with t4:
        st.caption("Sesión 4. Árbol multiplicativo con B constante y X que depende solo de S(T).")
        c = st.columns(5)
        S0 = num("S(0)", "w_f_S", 120.0, c[0], min_value=.001)
        u = num("u", "w_f_u", 1.7, c[1], min_value=.001)
        d_ = num("d", "w_f_d", .8, c[2], min_value=.001)
        N = int(num("N", "w_f_N", 3, c[3], min_value=1, max_value=200))
        B = num("B(,)", "w_f_B", float(np.exp(-.06)), c[4], min_value=.0001, format="%.6f")
        c = st.columns(2)
        X = txt("X(S(T))", "w_f_x", "max(S-K,0)", cont=c[0])
        K = num("K", "w_f_K", 115.0, c[1])
        r = seguro(cu.formula_binomial, S0, u, d_, N, B, X, K=K)
        if r:
            p, q, filas = r
            cc = st.columns(2)
            caja("X(0)", p, cont=cc[0])
            caja("q", q, secundaria=True, cont=cc[1])
            st.latex(r"X(0)=(B(\,,\,))^N\sum_{j=0}^N\binom Njq^j(1-q)^{N-j}X(S(0)u^jd^{N-j})")
            with st.expander("Detalle por número de subidas"):
                tabla(pd.DataFrame(filas))
            st.subheader("Prueba de consistencia")
            triv = cu.formula_binomial(S0, u, d_, N, B, "S")[0]
            fwd = cu.formula_binomial(S0, u, d_, N, B, "S-K", K=K)[0]
            cc = st.columns(2)
            caja("Derivado trivial X = S(T)", triv, f"debe dar S(0) = {S0:,.4f}", True, cont=cc[0])
            caja("Forward X = S(T) - K", fwd, f"debe dar S(0) - B(0,T)K = {S0 - B ** N * K:,.4f}", True, cont=cc[1])

    with t5:
        st.caption("Sesión 4. Del nodo j salen 2j y 2j+1. Sirve cuando ud ≠ du o B cambia con el nodo.")
        c = st.columns(3)
        S1 = num("S₁", "w_n_S1", 80.0, c[0], min_value=.001)
        S2 = num("S₂ (abajo)", "w_n_S2", 60.0, c[1], min_value=.001)
        S3 = num("S₃ (arriba)", "w_n_S3", 120.0, c[2], min_value=.001)
        c = st.columns(4)
        S4 = num("S₄ (dd)", "w_n_S4", 36.0, c[0], min_value=.001)
        S5 = num("S₅ (du)", "w_n_S5", 72.0, c[1], min_value=.001)
        S6 = num("S₆ (ud)", "w_n_S6", 80.0, c[2], min_value=.001)
        S7 = num("S₇ (uu)", "w_n_S7", 180.0, c[3], min_value=.001)
        c = st.columns(4)
        X4 = num("X₄", "w_n_X4", 0.0, c[0])
        X5 = num("X₅", "w_n_X5", 12.0, c[1])
        X6 = num("X₆", "w_n_X6", 20.0, c[2])
        X7 = num("X₇", "w_n_X7", 120.0, c[3])
        c = st.columns(3)
        b1 = num("B(0,1δ)", "w_n_b1", 1.0, c[0], min_value=.0001, step=.005)
        b2 = num("B(1δ,2δ) en nodo 2", "w_n_b2", 1.0, c[1], min_value=.0001, step=.005)
        b3 = num("B(1δ,2δ) en nodo 3", "w_n_b3", 1.0, c[2], min_value=.0001, step=.005)
        if S2 == S3 or S4 == S5 or S6 == S7:
            aviso("Revisa los nodos", "Los dos sucesores de un nodo deben tener precios distintos.")
        else:
            r = va.no_recombinante((S1, S2, S3, S4, S5, S6, S7), (b1, b2, b3), (X4, X5, X6, X7))
            c = st.columns(4)
            caja("X₁", r["V0"], cont=c[0])
            caja("q₁", r["q1"], secundaria=True, cont=c[1])
            caja("q₂", r["q2"], secundaria=True, cont=c[2])
            caja("q₃", r["q3"], secundaria=True, cont=c[3])
            st.latex(r"X_1=B(0,1\delta)B(1\delta,2\delta)\left[q_1q_3X_7+q_1(1-q_3)X_6+(1-q_1)q_2X_5+(1-q_1)(1-q_2)X_4\right]")
            st.caption("Esa forma cerrada aplica cuando B(1δ,2δ) es igual en los nodos 2 y 3. El valor de arriba descuenta nodo por nodo.")
