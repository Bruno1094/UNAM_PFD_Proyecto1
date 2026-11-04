import numpy as np
import pandas as pd
import streamlit as st

from motor import exoticos as ex
from motor import simulacion as si
from motor import vanilla as va
from paginas.comun import (K_T, aviso, caja, encabezado, figura, f4, graficar, lineas, lista_num, marca_v,
                           mercado, num, radio, sel, txt)


@st.cache_data(show_spinner=False, max_entries=64)
def _matriz(m, K, T, pasos, fechas, q_alt, D, dts, fix):
    base = m.con(q=0.0, pagos=())
    casos = {"Sin ingresos": base, f"Dividendo {q_alt:.2%}": base.con(q=q_alt),
             "Ingreso fijo": base.con(pagos=tuple((t, D) for t in dts))}
    filas = {}
    for nom, x in casos.items():
        col = {"Precio forward": float(x.forward(T))}
        for tp, et in (("call", "Call"), ("put", "Put")):
            col[f"{et} europeo"] = float(va.europea(x, K, T, tp))
            col[f"{et} Bermuda"] = va.precio_estilo(x, K, T, tp, "bermuda", pasos, fechas)
            col[f"{et} americano"] = va.precio_estilo(x, K, T, tp, "americana", pasos)
            col[f"{et} asiático"] = si.asiatica_mc(x, K, T, fix, tp, n=60_000).precio
        filas[nom] = col
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False, max_entries=32)
def _convergencia(m, K, T, tipo, estilo, tope):
    ns = np.unique(np.linspace(2, tope, 70).astype(int))
    return ns, [va.vanilla_arbol(m, K, T, tipo, estilo, int(x)).precio for x in ns]


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Calculadora de opciones", "Valúa un contrato con Black-Scholes o con árboles grandes, y compáralo entre estilos e ingresos.")
    t1, t2, t3, t4 = st.tabs(["Calculadora", "Comparar estilos e ingresos", "Convergencia", "Volatilidad implícita"])

    with t1:
        c = st.columns([1, 1.4, 1])
        tipo_txt = radio("Tipo", "w_op_tipo", ["Call", "Put"], cont=c[0])
        tipo = tipo_txt.lower()
        estilo = radio("Estilo", "w_op_estilo", ["Europea", "Americana", "Bermuda", "Asiática"], cont=c[1])
        pasos = num("Pasos del árbol", "w_op_N", 500, c[2], min_value=2, max_value=2000, step=50)
        if estilo == "Bermuda":
            f_txt = txt("Fechas de ejercicio (años, separadas por coma)", "w_op_fechas", "0.25, 0.5, 0.75")
            fechas = lista_num(f_txt)
            if not fechas:
                aviso("Revisa las fechas", "Escribe números separados por coma, por ejemplo 0.25, 0.5, 0.75.")
                return
        elif estilo == "Asiática":
            c = st.columns(3)
            promedio = radio("Promedio", "w_op_prom", ["Aritmético", "Geométrico"], cont=c[0])
            clase = radio("Promedio sustituye a", "w_op_clase", ["S_T (strike fijo)", "K (strike flotante)"], cont=c[1])
            fix = num("Fechas de observación", "w_op_fix", 12, c[2], min_value=1, max_value=365)

        eu = float(va.europea(m, K, T, tipo))
        if estilo == "Europea":
            d = va.detalle_bs(m, K, T)
            ar_v = va.vanilla_arbol(m, K, T, tipo, "europea", int(pasos)).precio
            c = st.columns(3)
            caja(f"{tipo_txt} europeo", eu, "Black-Scholes", cont=c[0])
            caja("Árbol binomial", ar_v, f"{int(pasos)} pasos", True, cont=c[1])
            caja("Paridad: c - p", eu - float(va.europea(m, K, T, "put" if tipo == "call" else "call")) if tipo == "call"
                 else float(va.europea(m, K, T, "call")) - eu, f"debe ser f = {float(m.valor_forward(K, T)):,.4f}", True, cont=c[2])
            with st.expander("Cómo se obtuvo", expanded=True):
                st.latex(rf"d_1=\frac{{\ln(S^*/K)+(r+\frac12\sigma^2)T}}{{\sigma\sqrt T}}={d['d1']:.6f},\quad d_2=d_1-\sigma\sqrt T={d['d2']:.6f}")
                if m.pagos or m.q:
                    st.latex(rf"\text{{Con ingresos, }}S\text{{ se reemplaza por el forward prepagado}}={d['P']:,.4f}")
                if tipo == "call":
                    st.latex(rf"c(0)=S\,N(d_1)-Ke^{{-rT}}N(d_2)={d['P']:,.4f}({d['Nd1']:.6f})-{d['KB']:,.4f}({d['Nd2']:.6f})={eu:,.4f}")
                else:
                    st.latex(rf"p(0)=Ke^{{-rT}}N(-d_2)-S\,N(-d_1)={d['KB']:,.4f}({d['N-d2']:.6f})-{d['P']:,.4f}({d['N-d1']:.6f})={eu:,.4f}")
        elif estilo in ("Americana", "Bermuda"):
            e_ = "americana" if estilo == "Americana" else "bermuda"
            fch = () if estilo == "Americana" else tuple(fechas)
            r = va.vanilla_arbol(m, K, T, tipo, e_, int(pasos), fch)
            p = va.precio_estilo(m, K, T, tipo, e_, int(pasos), fch)
            c = st.columns(3)
            caja(f"{tipo_txt} {estilo.lower()}", p, "árbol con variable de control", cont=c[0])
            caja("Prima por ejercicio anticipado", p - eu, "contra la europea", True, cont=c[1])
            caja("Árbol sin corrección", r.precio, f"{int(pasos)} pasos", True, cont=c[2])
            st.latex(r"h_k(y)=\max\left\{X(y),\;B\left[q\,h_{k+1}(uy)+(1-q)\,h_{k+1}(dy)\right]\right\}")
            if tipo == "call" and not m.q and not m.pagos:
                st.caption("Sin dividendos el call americano vale lo mismo que el europeo: nunca conviene ejercerlo antes.")
            if r.frontera:
                tt, ss = zip(*r.frontera)
                fig = figura("Frontera de ejercicio", "Tiempo (años)", "S crítico", 300)
                fig.add_scatter(x=tt, y=ss, mode="markers" if estilo == "Bermuda" else "lines",
                                line=dict(color="#2a62d6", width=2), marker=dict(size=9, color="#2a62d6"), showlegend=False)
                graficar(fig)
        else:
            flot = clase.startswith("K")
            if promedio == "Geométrico" and not flot and not m.pagos:
                p = ex.asiatica_geo_discreta(m, K, T, int(fix), tipo)
                c = st.columns(3)
                caja(f"{tipo_txt} asiático geométrico", p, "fórmula cerrada", cont=c[0])
                caja("Con monitoreo continuo", ex.asiatica_geo_continua(m, K, T, tipo), "Kemna-Vorst", True, cont=c[1])
                caja("Europeo equivalente", eu, secundaria=True, cont=c[2])
            elif promedio == "Aritmético":
                r = si.asiatica_mc(m, K, T, int(fix), tipo, "strike" if flot else "precio", 100_000)
                c = st.columns(3)
                caja(f"{tipo_txt} asiático aritmético", r.precio, "Monte Carlo, 100,000 trayectorias", cont=c[0])
                caja("Error estándar", r.ee, "con variable de control geométrica" if (not flot and not m.pagos) else "", True, cont=c[1])
                caja("Intervalo al 95 %", f"{r.ic[0]:,.4f} a {r.ic[1]:,.4f}", secundaria=True, cont=c[2])
            else:
                con = si.Contrato(base=tipo, K=K, referencia="final" if flot else "prom_geo",
                                  strike_tipo="prom_geo" if flot else "fijo")
                r = si.valuar_mc(con, m, T, int(fix), 100_000)
                c = st.columns(2)
                caja(f"{tipo_txt} asiático geométrico", r.precio, "Monte Carlo, 100,000 trayectorias", cont=c[0])
                caja("Error estándar", r.ee, secundaria=True, cont=c[1])
            st.latex(r"X=\left(A_T-K\right)_+\quad\text{o}\quad X=\left(S_T-A_T\right)_+,\qquad A_T=\frac1n\sum_{i=1}^n S(t_i)")
            st.caption("Aquí el promedio empieza en S(δ). En Árboles y Constructor puedes incluir S(0), como en el ejercicio 3 de la sesión 8.")

        s_x = np.linspace(.5 * min(m.S0, K), 1.5 * max(m.S0, K), 121)
        pago = np.maximum(s_x - K, 0) if tipo == "call" else np.maximum(K - s_x, 0)
        if m.pagos:
            actual = [float(va.europea(m.con(S0=float(x)), K, T, tipo)) for x in s_x]
        else:
            actual = va.europea(m.con(S0=s_x), K, T, tipo)
        fig = figura("Pago al vencimiento y valor europeo hoy", "Subyacente", "Valor")
        lineas(fig, s_x, {"Pago en T": pago, "Valor europeo hoy": actual})
        marca_v(fig, m.S0, "S₀")
        graficar(fig)

    with t2:
        st.caption("Los pagos fijos se reparten en las fechas que indiques. La columna de dividendos usa el rendimiento de abajo.")
        c = st.columns(5)
        q_alt = num("Rendimiento q (%)", "w_cmp_q", 2.0, c[0], step=.25, format="%.3f") / 100
        D = num("Pago fijo", "w_cmp_D", 2.0, c[1], min_value=0.0, step=.5)
        d_txt = txt("Fechas de pago", "w_cmp_f", "0.25, 0.75", cont=c[2])
        b_txt = txt("Fechas Bermuda", "w_cmp_b", "0.25, 0.5, 0.75", cont=c[3])
        fix = num("Promedio: fechas", "w_cmp_fix", 12, c[4], min_value=1, max_value=120)
        dts, fb = lista_num(d_txt), lista_num(b_txt)
        if dts is None or fb is None or not fb:
            aviso("Revisa las fechas", "Escribe números separados por coma.")
        else:
            with st.spinner("Calculando todas las combinaciones"):
                df = _matriz(m, K, T, 300, tuple(fb), q_alt, D, tuple(dts), int(fix))
            st.dataframe(df.style.format("{:,.4f}"), width="stretch")
            st.caption("Árboles de 300 pasos con variable de control; asiáticas aritméticas con 60,000 trayectorias.")
            aviso("Cómo leerla",
                  "Los ingresos bajan el precio del call y suben el del put, porque el título cae al pagarlos. "
                  "El call americano solo se separa del europeo cuando hay ingresos.", "neutro")

    with t3:
        c = st.columns(3)
        tp = radio("Tipo", "w_cv_t", ["Call", "Put"], cont=c[0]).lower()
        es = sel("Estilo", "w_cv_e", ["europea", "americana"], cont=c[1])
        tope = num("Pasos máximos", "w_cv_N", 200, c[2], min_value=10, max_value=800, step=10)
        ns, vs = _convergencia(m, K, T, tp, es, int(tope))
        fig = figura("Precio del árbol según el número de pasos", "Pasos N", "Precio")
        lineas(fig, ns, {"Árbol binomial": vs})
        if es == "europea":
            fig.add_hline(y=float(va.europea(m, K, T, tp)), line=dict(color="#d1344b", dash="dash", width=2),
                          annotation_text="Black-Scholes", annotation_font=dict(size=12))
        graficar(fig)

    with t4:
        c = st.columns(3)
        tp = radio("Tipo", "w_iv_t", ["Call", "Put"], cont=c[0]).lower()
        es = sel("Estilo", "w_iv_e", ["europea", "americana"], cont=c[1])
        pm = num("Precio de mercado", "w_iv_p", 10.0, c[2], min_value=0.0, step=.1)
        iv = va.vol_implicita(pm, m, K, T, tp, es)
        if iv is None:
            aviso("Ninguna volatilidad genera ese precio",
                  "El precio queda fuera del rango posible. Revísalo en la pestaña Arbitraje, sección Teórico contra mercado.")
        else:
            c = st.columns(2)
            caja("Volatilidad implícita", f"{iv:.4%}", cont=c[0])
            caja("Contra la σ capturada", f"{iv - m.sigma:+.4%}", "diferencia en puntos", True, cont=c[1])
