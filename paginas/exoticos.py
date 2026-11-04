import numpy as np
import streamlit as st

from motor import exoticos as ex
from motor import simulacion as si
from motor import vanilla as va
from motor.simulacion import Contrato
from paginas.comun import (K_T, aviso, caja, casilla, encabezado, graficar, figura, lista_num, mercado, num, radio,
                           sel, seguro, txt)

CATALOGO = {
    "Vistos en clase": ["Digital en efectivo", "Digital en activo", "Doble sin toque (banda)", "Lookback del máximo",
                        "Asiática aritmética"],
    "Otros de pago especial": ["Gap", "Supershare", "Potencia"],
    "Otras barreras": ["Barrera con rebate", "Doble barrera", "Un toque", "Sin toque", "Acumulador de rango"],
    "Otros dependientes de la trayectoria": ["Asiática geométrica", "Lookback flotante", "Lookback fijo"],
    "Decisiones en el tiempo": ["Selectora (chooser)", "Compuesta", "Inicio diferido", "Trinquete (cliquet)"],
    "Dos activos": ["Intercambio (Margrabe)", "Spread (Kirk)", "Mejor o peor de varios", "Canasta"],
}


def _tipo(k="w_ex_tipo", cont=None):
    return radio("Tipo", k, ["Call", "Put"], cont=cont).lower()


def _definir(nombre, m, K, T):
    """Dibuja los campos y regresa dict(precio, latex, contrato, pasos, resultado, nota, trayectoria)."""
    r = {"precio": None, "latex": "", "contrato": None, "pasos": 252, "resultado": None, "nota": "", "tray": False}
    c = st.columns(3)
    if nombre == "Digital en efectivo":
        tp = _tipo(cont=c[0]); monto = num("Monto", "w_ex_mon", 10.0, c[1], min_value=0.0)
        r["precio"] = float(ex.digital_efectivo(m, K, T, tp, monto))
        r["latex"] = r"V=\text{monto}\cdot B(0,T)\,N(\pm d_2)"
        r["contrato"] = Contrato(base="digital_call" if tp == "call" else "digital_put", K=K, monto=monto); r["pasos"] = 1
    elif nombre == "Digital en activo":
        tp = _tipo(cont=c[0])
        r["precio"] = float(ex.digital_activo(m, K, T, tp))
        r["latex"] = r"V=S_0B_E(0,T)\,N(\pm d_1)"
        if tp == "call":
            r["contrato"] = Contrato(base="digital_activo", K=K); r["pasos"] = 1
        else:
            r["contrato"] = Contrato(base="expresion", expresion="S*(S<K)", K=K); r["pasos"] = 1
    elif nombre == "Gap":
        tp = _tipo(cont=c[0]); K2 = num("Strike de pago K₂", "w_ex_K2", 90.0, c[1], min_value=0.0001)
        r["precio"] = float(ex.gap(m, K, K2, T, tp))
        r["latex"] = r"X=(S_T-K_2)\,\mathbb 1\{S_T>K\}\;\Rightarrow\;V=S_0B_E\,N(d_1)-K_2B\,N(d_2)"
        ex_ = "(S-K2)*(S>K)" if tp == "call" else "(K2-S)*(S<K)"
        r["contrato"] = Contrato(base="expresion", expresion=ex_, K=K, K2=K2); r["pasos"] = 1
    elif nombre == "Supershare":
        K2 = num("Límite superior K₂", "w_ex_ss", 120.0, c[0], min_value=0.0001)
        if K2 <= K:
            raise ValueError("K₂ debe ser mayor que K.")
        r["precio"] = float(ex.supershare(m, K, K2, T))
        r["latex"] = r"X=\frac{S_T}{K}\,\mathbb 1\{K<S_T<K_2\}"
        r["contrato"] = Contrato(base="expresion", expresion="S/K*(S>K)*(S<K2)", K=K, K2=K2); r["pasos"] = 1
    elif nombre == "Potencia":
        tp = _tipo(cont=c[0]); a = num("Exponente a", "w_ex_a", 2.0, c[1], min_value=.1, step=.5)
        Kp = num("Strike sobre S^a", "w_ex_Kp", K ** 2 if K < 1000 else K, c[2], min_value=.0001)
        r["precio"] = float(ex.potencia(m, Kp, T, a, tp))
        r["latex"] = r"X=(S_T^{a}-K)_+"
        r["contrato"] = Contrato(base=tp, K=Kp, potencia=a); r["pasos"] = 1
    elif nombre == "Barrera con rebate":
        tp = _tipo(cont=c[0]); d = radio("Dirección", "w_ex_dir", ["Abajo", "Arriba"], cont=c[1]).lower()
        ac = radio("Efecto", "w_ex_ac", ["Sale", "Entra"], cont=c[2]).lower()
        c = st.columns(3)
        H = num("Barrera H", "w_ex_H", .85 * m.S0 if d == "abajo" else 1.2 * m.S0, c[0], min_value=.0001)
        R = num("Rebate", "w_ex_R", 0.0, c[1], min_value=0.0)
        r["tray"] = True
        r["precio"] = ex.barrera(m, K, H, T, tp, d, ac, R) if not m.pagos else None
        r["latex"] = r"\text{in}+\text{out}=\text{vainilla}\quad\text{(Reiner-Rubinstein)}"
        r["contrato"] = Contrato(base=tp, K=K, barrera=ac, L=H if d == "abajo" else None, U=H if d == "arriba" else None,
                                 rebate=R, rebate_momento="toque" if ac == "sale" else "vencimiento", bgk=True)
        r["nota"] = "Rebate: en las de salida se paga al tocar; en las de entrada, en T si nunca entró."
    elif nombre == "Doble barrera":
        tp = _tipo(cont=c[0]); ac = radio("Efecto", "w_ex_ac2", ["Sale", "Entra"], cont=c[1]).lower()
        c = st.columns(2)
        L = num("Banda inferior L", "w_ex_L", .8 * m.S0, c[0], min_value=.0001)
        U = num("Banda superior U", "w_ex_U", 1.25 * m.S0, c[1], min_value=.0001)
        r["tray"] = True
        r["precio"] = ex.doble_barrera(m, K, L, U, T, tp, ac) if not m.pagos else None
        r["latex"] = r"V=B(0,T)\int_L^U X(s)\,p_{banda}(s)\,ds\quad\text{(densidad por imágenes)}"
        r["contrato"] = Contrato(base=tp, K=K, barrera=ac, L=L, U=U, bgk=True)
    elif nombre == "Doble sin toque (banda)":
        L = num("B₁", "w_ex_B1", .8 * m.S0, c[0], min_value=.0001)
        U = num("B₂", "w_ex_B2", 1.25 * m.S0, c[1], min_value=.0001)
        mon = num("Pago", "w_ex_mb", float(m.S0), c[2], min_value=0.0)
        r["tray"] = True
        r["precio"] = ex.opcion_banda(m, L, U, T, mon) if not m.pagos else None
        r["latex"] = r"X=\text{pago}\cdot\mathbb 1\{B_1<S(t)<B_2,\ \forall t\in[0,T]\}"
        r["contrato"] = Contrato(base="monto", monto=mon, barrera="sale", L=L, U=U, bgk=True)
        r["nota"] = "La fórmula vigila continuamente. Para valuarla en un árbol como en clase, usa la pestaña Árboles."
    elif nombre == "Un toque":
        H = num("Nivel H", "w_ex_Ht", .85 * m.S0, c[0], min_value=.0001)
        mon = num("Monto", "w_ex_mt", 10.0, c[1], min_value=0.0)
        mo = radio("Se paga", "w_ex_mom", ["Al tocar", "Al vencimiento"], cont=c[2])
        mm = "toque" if mo == "Al tocar" else "vencimiento"
        r["tray"] = True
        r["precio"] = ex.un_toque(m, H, T, mon, mm) if not m.pagos else None
        r["latex"] = r"X=\text{monto}\cdot\mathbb 1\{\tau_H\le T\}"
        r["contrato"] = Contrato(base="monto", monto=0, barrera="sale", L=H if H < m.S0 else None, U=H if H >= m.S0 else None,
                                 rebate=mon, rebate_momento=mm, bgk=True)
    elif nombre == "Sin toque":
        H = num("Nivel H", "w_ex_Hn", .85 * m.S0, c[0], min_value=.0001)
        mon = num("Monto", "w_ex_mn", 10.0, c[1], min_value=0.0)
        r["tray"] = True
        r["precio"] = ex.sin_toque(m, H, T, mon) if not m.pagos else None
        r["latex"] = r"X=\text{monto}\cdot\mathbb 1\{\tau_H>T\}"
        r["contrato"] = Contrato(base="monto", monto=mon, barrera="sale", L=H if H < m.S0 else None, U=H if H >= m.S0 else None, bgk=True)
    elif nombre == "Acumulador de rango":
        L = num("Límite inferior", "w_ex_rL", .9 * m.S0, c[0], min_value=.0001)
        U = num("Límite superior", "w_ex_rU", 1.1 * m.S0, c[1], min_value=.0001)
        cup = num("Cupón total", "w_ex_rc", 10.0, c[2], min_value=0.0)
        r["tray"] = True
        r["precio"] = ex.acumulador_rango(m, L, U, T, cup) if not m.pagos else None
        r["latex"] = r"X=\text{cupón}\cdot\frac1n\sum_{i=1}^n\mathbb 1\{L\le S(t_i)\le U\}"
        r["contrato"] = Contrato(base="monto", monto=0, rango_L=L, rango_U=U, rango_cupon=cup)
    elif nombre == "Asiática geométrica":
        tp = _tipo(cont=c[0]); fx = num("Fechas de observación", "w_ex_fg", 12, c[1], min_value=1, max_value=365)
        r["tray"] = True
        r["precio"] = ex.asiatica_geo_discreta(m, K, T, int(fx), tp) if not m.pagos else None
        r["latex"] = r"X=\left(\Big[\prod_i S(t_i)\Big]^{1/n}-K\right)_+"
        r["contrato"] = Contrato(base=tp, K=K, referencia="prom_geo"); r["pasos"] = int(fx)
    elif nombre == "Asiática aritmética":
        tp = _tipo(cont=c[0]); fx = num("Fechas de observación", "w_ex_fa", 12, c[1], min_value=1, max_value=365)
        r["resultado"] = si.asiatica_mc(m, K, T, int(fx), tp, n=100_000)
        r["latex"] = r"X=\left(\frac1n\sum_iS(t_i)-K\right)_+"
    elif nombre == "Lookback flotante":
        tp = _tipo(cont=c[0])
        r["tray"] = True
        r["precio"] = ex.lookback_flotante(m, T, tp) if not m.pagos else None
        r["latex"] = r"\text{Call: }S_T-\min_tS(t)\qquad\text{Put: }\max_tS(t)-S_T"
        r["contrato"] = Contrato(base="call" if tp == "call" else "put", referencia="final",
                                 strike_tipo="min" if tp == "call" else "max", bgk=False)
        r["nota"] = "La simulación observa en fechas discretas, por eso da un valor algo menor que la fórmula continua."
    elif nombre == "Lookback fijo":
        tp = _tipo(cont=c[0])
        r["tray"] = True
        r["precio"] = ex.lookback_fijo(m, K, T, tp) if not m.pagos else None
        r["latex"] = r"\text{Call: }(\max_tS(t)-K)_+\qquad\text{Put: }(K-\min_tS(t))_+"
        r["contrato"] = Contrato(base=tp, K=K, referencia="max" if tp == "call" else "min")
        r["nota"] = "La simulación observa en fechas discretas, por eso da un valor algo menor que la fórmula continua."
    elif nombre == "Lookback del máximo":
        Nn = num("Pasos del árbol (N, máximo 16)", "w_ex_lkN", 8, c[0], min_value=1, max_value=16)
        r["tray"] = True
        arb = si.valuar_arbol(Contrato(base="expresion", expresion="M"), m, T, int(Nn))
        r["extra"] = ("Árbol exacto con N pasos", arb.precio)
        r["precio"] = ex.lookback_maximo(m, T) if not m.pagos else None
        r["latex"] = r"X=\max\{S(0),S(\delta),\dots,S(N\delta)\}"
        r["contrato"] = Contrato(base="expresion", expresion="M", K=K)
        r["nota"] = "La fórmula supone monitoreo continuo; el árbol observa solo en sus N fechas."
    elif nombre == "Selectora (chooser)":
        te = num("Fecha de elección", "w_ex_te", T / 2, c[0], min_value=.001, max_value=float(T))
        r["precio"] = ex.selectora(m, K, te, T)
        r["latex"] = r"V_{t_e}=\max\{c(S,K,T-t_e),\;p(S,K,T-t_e)\}"
        r["nota"] = "Comprobada contra el valor publicado de Haug." + (" Con pagos discretos se aproxima con S₀ - C." if m.pagos else "")
    elif nombre == "Compuesta":
        ext = radio("Exterior", "w_ex_ce", ["Call", "Put"], cont=c[0]).lower()
        inte = radio("Sobre un", "w_ex_ci", ["Call", "Put"], cont=c[1]).lower()
        c = st.columns(2)
        t1 = num("Vencimiento exterior t₁", "w_ex_t1", T / 2, c[0], min_value=.001, max_value=float(T))
        Ke = num("Strike exterior", "w_ex_ke", 5.0, c[1], min_value=0.0)
        r["precio"] = ex.compuesta(m, Ke, t1, K, T, ext, inte)
        r["latex"] = r"V=e^{-rt_1}\,E_Q\left[(V_{int}(S_{t_1})-K_{ext})_+\right]\quad\text{(Geske, por integral numérica)}"
        r["nota"] = "Comprobada contra el valor publicado de Haug y contra su paridad." + (" Con pagos discretos se aproxima con S₀ - C." if m.pagos else "")
    elif nombre == "Inicio diferido":
        tp = _tipo(cont=c[0])
        ti = num("Fecha de inicio", "w_ex_ti", T / 2, c[1], min_value=.001, max_value=float(T) * .999)
        al = num("K como proporción de S(t)", "w_ex_al", 1.0, c[2], min_value=.01, step=.05)
        r["tray"] = True
        r["precio"] = ex.inicio_diferido(m, al, ti, T, tp) if not m.pagos else None
        r["latex"] = r"V=S_0e^{-qt}\,\mathrm{BS}(1,\alpha,T-t)"
        r["contrato"] = Contrato(base=tp, strike_tipo="adelantado", t_adelantado=ti, alfa_adelantado=al)
    elif nombre == "Trinquete (cliquet)":
        pe = num("Periodos", "w_ex_pe", 4, c[0], min_value=1, max_value=60)
        nomi = num("Nominal", "w_ex_no", 100.0, c[1], min_value=0.0)
        tl = num("Tope por periodo (%), 0 sin tope", "w_ex_tl", 0.0, c[2], min_value=0.0, step=1.0, format="%.2f")
        c = st.columns(2)
        pg = num("Piso global (%), 0 sin piso", "w_ex_pg", 0.0, c[0], min_value=0.0, step=1.0, format="%.2f")
        tg = num("Tope global (%), 0 sin tope", "w_ex_tg", 0.0, c[1], min_value=0.0, step=1.0, format="%.2f")
        r["resultado"] = si.trinquete_mc(m, T, int(pe), tl / 100 if tl else None, 0.0, pg / 100 if pg else None,
                                         tg / 100 if tg else None, nomi)
        if not (tl or pg or tg) and not m.pagos:
            r["precio"] = nomi * ex.trinquete(m, T, int(pe))
        r["latex"] = r"X=\text{nominal}\cdot\sum_i\left(\min\{R_i,\text{tope}\}\right)_+,\quad R_i=\frac{S(t_{i+1})}{S(t_i)}-1"
    elif nombre in ("Intercambio (Margrabe)", "Spread (Kirk)", "Mejor o peor de varios", "Canasta"):
        return _multi(nombre, m, K, T, r, c)
    return r


def _multi(nombre, m, K, T, r, c):
    S2 = num("S₂", "w_ex_S2", m.S0, c[0], min_value=.0001)
    s2 = num("σ₂ (%)", "w_ex_s2", 25.0, c[1], min_value=.01, format="%.2f") / 100
    rho = num("Correlación ρ", "w_ex_rho", .5, c[2], min_value=-.99, max_value=.99, step=.05)
    S1, s1, q1, q2 = m.S0, m.sigma, m.qc, m.qc
    if nombre == "Intercambio (Margrabe)":
        r["precio"] = ex.margrabe(S1, S2, q1, q2, s1, s2, rho, T)
        r["resultado"] = si.multiactivo_mc([S1, S2], [s1, s2], [q1, q2], m.rc, rho, T, lambda s: np.maximum(s[:, 0] - s[:, 1], 0))
        r["latex"] = r"X=(S_1-S_2)_+,\quad\sigma^2=\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2"
    elif nombre == "Spread (Kirk)":
        Ks = num("Strike del spread", "w_ex_Ks", 5.0, min_value=0.0)
        r["precio"] = ex.kirk(S1, S2, Ks, m.rc, q1, q2, s1, s2, rho, T)
        r["resultado"] = si.multiactivo_mc([S1, S2], [s1, s2], [q1, q2], m.rc, rho, T, lambda s: np.maximum(s[:, 0] - s[:, 1] - Ks, 0))
        r["latex"] = r"X=(S_1-S_2-K)_+\quad\text{(aproximación de Kirk)}"
    elif nombre == "Mejor o peor de varios":
        tipo = radio("Pago", "w_ex_bw", ["Mejor de dos", "Peor de dos"])
        fn = np.max if tipo.startswith("Mejor") else np.min
        r["resultado"] = si.multiactivo_mc([S1, S2], [s1, s2], [q1, q2], m.rc, rho, T, lambda s: np.maximum(fn(s, axis=1) - K, 0))
        r["latex"] = r"X=(\max\{S_1,S_2\}-K)_+\quad\text{o}\quad(\min\{S_1,S_2\}-K)_+"
    else:
        w = num("Peso del activo 1", "w_ex_w", .5, min_value=0.0, max_value=1.0, step=.05)
        tp = _tipo("w_ex_tcan")
        sg = 1 if tp == "call" else -1
        r["resultado"] = si.multiactivo_mc([S1, S2], [s1, s2], [q1, q2], m.rc, rho, T,
                                           lambda s: np.maximum(sg * (w * s[:, 0] + (1 - w) * s[:, 1] - K), 0))
        r["latex"] = r"X=(wS_1+(1-w)S_2-K)_+"
    return r


@st.cache_data(show_spinner=False, max_entries=64)
def _mc(_c, firma, m, T, pasos, n):
    r = si.valuar_mc(_c, m, T, pasos, n)
    r.vp = None
    return r


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Exóticos", "Primero están los que vimos en clase. Cada contrato muestra su pago, su valor y una comprobación por simulación.")
    c = st.columns(2)
    fam = sel("Familia", "w_ex_fam", list(CATALOGO), cont=c[0])
    nombre = sel("Contrato", f"w_ex_nom_{list(CATALOGO).index(fam)}", CATALOGO[fam], cont=c[1])

    r = seguro(_definir, nombre, m, K, T)
    if r is None:
        return
    if r["latex"]:
        st.latex(r["latex"])
    res, prec = r["resultado"], r["precio"]
    c = st.columns(3)
    if prec is not None:
        caja(nombre, float(prec), "fórmula cerrada", cont=c[0])
    if res is not None:
        caja("Monte Carlo" if prec is not None else nombre, res.precio, f"error estándar {res.ee:,.4f}", prec is not None, cont=c[1 if prec is not None else 0])
    if "extra" in r:
        caja(r["extra"][0], r["extra"][1], secundaria=True, cont=c[2])
    if prec is None and res is None and r["tray"]:
        aviso("Con pagos discretos la fórmula cerrada no aplica",
              "La barrera o la trayectoria se observan sobre S, no sobre S₀ - C. Usa la simulación de abajo.", "modelo")

    if r["contrato"] is not None:
        st.subheader("Comprobar con simulación")
        c1, c2 = st.columns([1, 2])
        pasos = r["pasos"]
        n = sel("Trayectorias", "w_ex_n", [20_000, 50_000, 100_000], default=50_000, cont=c1,
                format_func=lambda x: f"{x:,}")
        firma = repr(r["contrato"])
        if c1.button("Simular", key="ex_sim", type="primary" if prec is None else "secondary"):
            with st.spinner("Simulando"):
                try:
                    v = _mc(r["contrato"], firma, m, T, int(pasos), int(n))
                except ValueError as e:
                    aviso("No pude simularlo", str(e))
                    return
                st.session_state["_ex_res"] = (firma, m, T, v.precio, v.ee, v.ic)
        g = st.session_state.get("_ex_res")
        if g and g[0] == firma and g[1] == m and g[2] == T:
            _, _, _, p, ee, ic = g
            pie = f"intervalo al 95 %: {ic[0]:,.4f} a {ic[1]:,.4f}"
            caja("Monte Carlo", p, pie, cont=c2)
            if prec is not None and ee > 0:
                z = abs(p - float(prec)) / ee
                c2.caption(f"Diferencia contra la fórmula: {p - float(prec):+,.4f} ({z:.1f} errores estándar).")
    if r["nota"]:
        st.caption(r["nota"])
