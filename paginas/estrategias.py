import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from motor import vanilla as va
from paginas.comun import (K_T, aviso, caja, encabezado, figura, graficar, lineas, marca_v, mercado, num, reemplazar_tabla,
                           sel, tabla_editable)

TIPOS = ["Call", "Put", "Forward", "Subyacente"]


def _plantillas(K, T):
    f = lambda filas: pd.DataFrame(filas, columns=["Instrumento", "Cantidad", "Strike", "Plazo"])
    a, b = round(.9 * K, 2), round(1.1 * K, 2)
    return {
        "Call largo": f([["Call", 1, K, T]]),
        "Straddle": f([["Call", 1, K, T], ["Put", 1, K, T]]),
        "Strangle": f([["Put", 1, a, T], ["Call", 1, b, T]]),
        "Call spread": f([["Call", 1, K, T], ["Call", -1, b, T]]),
        "Mariposa": f([["Call", 1, a, T], ["Call", -2, K, T], ["Call", 1, b, T]]),
        "Collar": f([["Subyacente", 1, 0, T], ["Put", 1, a, T], ["Call", -1, b, T]]),
        "Put protector": f([["Subyacente", 1, 0, T], ["Put", 1, a, T]]),
        "Call cubierto": f([["Subyacente", 1, 0, T], ["Call", -1, b, T]]),
        "Sintético": f([["Call", 1, K, T], ["Put", -1, K, T]]),
    }


def valor(m, pos, S=None, trans=0.0, dvol=0.0):
    """Valor de la cartera con S y volatilidad movidos, trans años después."""
    S = m.S0 if S is None else S
    mm = m.avanzar(trans).con(S0=S, sigma=max(m.sigma + dvol, 1e-4))
    total = 0.0
    for tipo, q, k, t in pos:
        t_rest = max(t - trans, 1e-8)
        if tipo == "Subyacente":
            total = total + q * S
        elif tipo == "Forward":
            total = total + q * mm.valor_forward(k, t_rest)
        else:
            total = total + q * va.europea(mm, k, t_rest, tipo.lower())
    return total


def pago_final(pos, S):
    t = np.zeros_like(S)
    for tipo, q, k, _ in pos:
        if tipo == "Call":
            t = t + q * np.maximum(S - k, 0)
        elif tipo == "Put":
            t = t + q * np.maximum(k - S, 0)
        elif tipo == "Forward":
            t = t + q * (S - k)
        else:
            t = t + q * S
    return t


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Estrategias y riesgo", "Complemento: no viene en las sesiones. Arma una cartera de opciones, forwards y títulos, y mide escenarios y VaR.")
    c = st.columns([2, 1])
    plantillas = _plantillas(K, T)
    p = c[0].selectbox("Plantilla", list(plantillas), index=1, key="e_pl")
    if c[1].button("Cargar plantilla", width="stretch"):
        reemplazar_tabla("cartera", plantillas[p])
    df = tabla_editable("cartera", plantillas["Straddle"], num_rows="dynamic", hide_index=True, width="stretch",
                        column_config={"Instrumento": st.column_config.SelectboxColumn(options=TIPOS, required=True),
                                       "Cantidad": st.column_config.NumberColumn(help="Negativa si es posición corta"),
                                       "Strike": st.column_config.NumberColumn(min_value=0.0),
                                       "Plazo": st.column_config.NumberColumn("Plazo (años)", min_value=.003)})
    lim = df.dropna()
    pos = [(a, float(b), float(k), float(t)) for a, b, k, t in zip(lim["Instrumento"], lim["Cantidad"], lim["Strike"], lim["Plazo"])]
    if not pos:
        aviso("La cartera está vacía", "Agrega al menos una fila o carga una plantilla.")
        return
    v0 = float(valor(m, pos))
    h = .01 * m.S0
    delta = float((valor(m, pos, m.S0 + h) - valor(m, pos, m.S0 - h)) / (2 * h))
    gamma = float((valor(m, pos, m.S0 + h) - 2 * v0 + valor(m, pos, m.S0 - h)) / h ** 2)
    vega = float((valor(m, pos, dvol=.01) - valor(m, pos, dvol=-.01)) / 2)
    theta = float(valor(m, pos, trans=1 / 365) - v0)
    c = st.columns(5)
    caja("Costo hoy", v0, cont=c[0])
    caja("Delta", delta, secundaria=True, cont=c[1])
    caja("Gamma", gamma, secundaria=True, cont=c[2])
    caja("Vega", vega, "por 1 punto de σ", True, cont=c[3])
    caja("Theta", theta, "por día", True, cont=c[4])

    t1, t2, t3 = st.tabs(["Pago", "Escenarios", "VaR"])
    with t1:
        ks = [k for a, _, k, _ in pos if k > 0] + [m.S0]
        xs = np.linspace(.6 * min(ks), 1.4 * max(ks), 241)
        pagos = pago_final(pos, xs)
        gan = pagos - v0 / float(m.B(max(x[3] for x in pos)))
        fig = figura("Resultado al vencimiento", "S_T", "Valor")
        lineas(fig, xs, {"Pago": pagos, "Ganancia neta del costo capitalizado": gan})
        marca_v(fig, m.S0, "S₀")
        fig.add_hline(y=0, line=dict(color="#7D8AA3", width=1))
        graficar(fig)
        signo = np.sign(gan)
        cortes = xs[1:][signo[1:] * signo[:-1] < 0]
        if len(cortes):
            st.caption("Puntos de equilibrio: " + ", ".join(f"{x:,.2f}" for x in cortes))
        if len({x[3] for x in pos}) > 1:
            st.caption("Hay plazos distintos: la gráfica supone que todo vence en el plazo mayor.")

    with t2:
        st.caption("Ganancia o pérdida inmediata al mover subyacente y volatilidad, revaluando la cartera.")
        cS = np.array([-.2, -.15, -.1, -.05, 0, .05, .1, .15, .2])
        cV = np.array([-.1, -.05, 0, .05, .1])
        Z = np.array([[float(valor(m, pos, m.S0 * (1 + a), dvol=b)) - v0 for a in cS] for b in cV])
        lim_ = max(abs(Z.min()), abs(Z.max())) or 1
        fig = figura("Ganancia o pérdida por escenario", "Movimiento del subyacente", "Movimiento de σ", 340)
        fig.add_trace(go.Heatmap(x=[f"{a:+.0%}" for a in cS], y=[f"{b * 100:+.0f} pts" for b in cV], z=Z, colorscale="RdBu", zmid=0,
                                 zmin=-lim_, zmax=lim_, text=np.round(Z, 2), texttemplate="%{text}", textfont=dict(size=11),
                                 hovertemplate="S %{x}, σ %{y}<br>%{z:,.4f}<extra></extra>", colorbar=dict(thickness=12)))
        fig.update_layout(hovermode="closest")
        graficar(fig)

    with t3:
        c = st.columns(3)
        dias = num("Horizonte (días)", "w_e_dias", 10, c[0], min_value=1, max_value=250)
        conf = sel("Confianza", "w_e_conf", [.95, .975, .99], default=.99, cont=c[1], format_func=lambda x: f"{x:.1%}")
        mu = num("Media anual de S (%)", "w_e_mu", 0.0, c[2], step=1.0, format="%.2f") / 100
        hz = dias / 365
        if hz >= min(x[3] for x in pos):
            aviso("El horizonte es mayor que el plazo más corto", "Reduce el horizonte para poder revaluar la cartera.")
            return
        rng = np.random.default_rng(11)
        Z = rng.standard_normal(100_000)
        ST = m.S0 * np.exp((mu - .5 * m.sigma ** 2) * hz + m.sigma * np.sqrt(hz) * Z)
        if m.pagos:
            pl = np.array([float(valor(m, pos, float(s), trans=hz)) for s in ST[:4000]]) - v0
        else:
            pl = np.asarray(valor(m, pos, ST, trans=hz)) - v0
        var_c = -np.quantile(pl, 1 - conf)
        cola = pl[pl <= -var_c]
        es = -cola.mean() if len(cola) else var_c
        from scipy.stats import norm
        z = norm.ppf(conf)
        d_S = m.S0 * m.sigma * np.sqrt(hz)
        var_d = z * abs(delta) * d_S
        var_dg = -np.quantile(delta * (ST - m.S0) + .5 * gamma * (ST - m.S0) ** 2, 1 - conf)
        c = st.columns(4)
        caja(f"VaR {conf:.1%}", float(var_c), "revaluación completa", cont=c[0])
        caja("Déficit esperado", float(es), "promedio de la cola", True, cont=c[1])
        caja("VaR delta", float(var_d), "aproximación lineal", True, cont=c[2])
        caja("VaR delta-gamma", float(var_dg), "aproximación cuadrática", True, cont=c[3])
        fig = figura("Distribución de ganancias y pérdidas en el horizonte", "Ganancia o pérdida", "Escenarios", 300)
        fig.add_histogram(x=pl, nbinsx=70, marker_color="#2a62d6", showlegend=False)
        fig.add_vline(x=-var_c, line=dict(color="#d1344b", width=2, dash="dash"), annotation_text="VaR", annotation_font=dict(size=12))
        fig.update_layout(hovermode="closest", bargap=.03)
        graficar(fig)
        st.caption("Simulación de 100,000 escenarios lognormales de S con σ constante. La volatilidad y las tasas no se mueven.")
