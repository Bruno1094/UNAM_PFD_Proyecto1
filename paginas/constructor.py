import numpy as np
import plotly.graph_objects as go
import streamlit as st

from motor import simulacion as si
from motor.simulacion import Contrato
from paginas.comun import (K_T, SERIES, aviso, caja, casilla, encabezado, figura, graficar, lista_num, mercado, num,
                           radio, sel, txt)

BASES = {"Call": "call", "Put": "put", "Straddle": "straddle", "Digital call": "digital_call", "Digital put": "digital_put",
         "Digital en activo": "digital_activo", "Forward": "forward", "Monto fijo (sin strike)": "monto",
         "Call spread": "call_spread", "Put spread": "put_spread", "Fórmula propia": "expresion"}
REFS = {"S_T": "final", "Promedio aritmético": "prom_ar", "Promedio geométrico": "prom_geo", "Máximo": "max", "Mínimo": "min"}
STRIKES = {"Fijo": "fijo", "Promedio aritmético": "prom_ar", "Promedio geométrico": "prom_geo", "Máximo": "max",
           "Mínimo": "min", "Se fija después (inicio diferido)": "adelantado"}
BARS = ["Sin barrera", "Inferior", "Superior", "Banda (inferior y superior)"]


def _preset(nombre):
    s = st.session_state
    m, K = s["mercado"], s["K"]
    S = m.S0
    limpio = {"w_c_base": "Call", "w_c_ref": "S_T", "w_c_str": "Fijo", "w_c_bar": "Sin barrera", "w_c_efecto": "Sale",
              "w_c_reb": 0.0, "w_c_tope_on": False, "w_c_piso_on": False, "w_c_auto": False, "w_c_rango": False,
              "w_c_ej": "Europeo", "w_c_mon": 10.0, "w_c_part": 1.0, "w_c_pot": 1.0, "w_c_nom": 1.0}
    cambios = {
        "Doble sin toque de clase": {"w_c_base": "Monto fijo (sin strike)", "w_c_mon": float(S), "w_c_bar": "Banda (inferior y superior)",
                                         "w_c_L": .5 * S, "w_c_U": 2.5 * S},
        "Lookback de clase": {"w_c_base": "Fórmula propia", "w_c_expr": "M"},
        "Asiática con barrera y rebate": {"w_c_ref": "Promedio aritmético", "w_c_bar": "Superior", "w_c_U": 1.3 * S, "w_c_reb": 2.0},
        "Put americano con barrera": {"w_c_base": "Put", "w_c_bar": "Inferior", "w_c_L": .7 * S, "w_c_ej": "Americano"},
        "Call con tope y participación": {"w_c_part": 1.5, "w_c_tope_on": True, "w_c_tope": .3 * S},
        "Nota autollamable": {"w_c_base": "Monto fijo (sin strike)", "w_c_mon": 100.0, "w_c_auto": True, "w_c_a_niv": float(S),
                              "w_c_a_cup": 4.0, "w_c_a_f": "0.25, 0.5, 0.75"},
        "Acumulador en banda con salida": {"w_c_base": "Monto fijo (sin strike)", "w_c_mon": 0.0, "w_c_rango": True, "w_c_r_L": .9 * S,
                                           "w_c_r_U": 1.1 * S, "w_c_r_c": 10.0, "w_c_bar": "Banda (inferior y superior)", "w_c_L": .75 * S, "w_c_U": 1.3 * S},
    }
    s.update(limpio)
    s.update(cambios.get(nombre, {}))
    s.pop("_c_res", None)


def _formulario(m, K, T):
    izq, der = st.columns(2)
    with izq:
        st.subheader("Pago")
        c = st.columns(2)
        base = BASES[sel("Tipo de pago", "w_c_base", list(BASES), cont=c[0])]
        ref = REFS[sel("Se mide sobre", "w_c_ref", list(REFS), cont=c[1], disabled=base in ("monto", "expresion"))]
        expr = ""
        if base == "expresion":
            expr = txt("Pago", "w_c_expr", "max(S-K,0)*si(A>K,1,0)",
                       help="Variables: S (final), A (promedio), G (geométrico), M (máximo), m (mínimo), K, K2, S0, T. "
                            "Funciones: max, min, abs, exp, log, sqrt, si(condición, a, b).")
            st.caption("Variables: S, A, G, M, m, K, K2, S0, T. Funciones: max, min, abs, exp, log, sqrt, si(c, a, b).")
        c = st.columns(2)
        st_tipo = STRIKES[sel("Strike", "w_c_str", list(STRIKES), cont=c[0],
                              disabled=base in ("monto", "expresion"))]
        K1 = num("K", "w_c_K", float(K), c[1], min_value=0.0)
        K2, monto, t_ad, al_ad = 0.0, 1.0, 0.0, 1.0
        if base in ("call_spread", "put_spread", "expresion"):
            K2 = num("K₂", "w_c_K2", 1.15 * float(K), min_value=0.0)
        if base in ("digital_call", "digital_put", "monto"):
            monto = num("Monto", "w_c_mon", 10.0, min_value=0.0)
        if st_tipo == "adelantado":
            c = st.columns(2)
            t_ad = num("Fecha en que se fija", "w_c_tad", T / 2, c[0], min_value=0.0, max_value=float(T))
            al_ad = num("K como proporción de S(t)", "w_c_alad", 1.0, c[1], min_value=.01, step=.05)
        t_prom, con_S0 = 0.0, False
        if ref in ("prom_ar", "prom_geo") or st_tipo in ("prom_ar", "prom_geo"):
            c = st.columns(2)
            t_prom = num("El promedio empieza en", "w_c_tp", 0.0, c[0], min_value=0.0, max_value=float(T))
            con_S0 = casilla("Incluir S(0) en el promedio, como en clase", "w_c_ps0", True, cont=c[1])
        c = st.columns(3)
        part = num("Participación", "w_c_part", 1.0, c[0], step=.1)
        pot = num("Potencia de S", "w_c_pot", 1.0, c[1], min_value=.1, step=.5)
        nom = num("Nominal", "w_c_nom", 1.0, c[2], min_value=0.0)
        c = st.columns(2)
        tope = piso = None
        if casilla("Agregar tope", "w_c_tope_on", cont=c[0]):
            tope = num("Tope", "w_c_tope", .3 * m.S0, c[0])
        if casilla("Agregar piso", "w_c_piso_on", cont=c[1]):
            piso = num("Piso", "w_c_piso", 0.0, c[1])

        st.subheader("Ejercicio")
        ej = radio("Se puede ejercer", "w_c_ej", ["Europeo", "Americano", "Bermuda"])
        fechas = ()
        if ej == "Bermuda":
            fechas = tuple(lista_num(txt("Fechas de ejercicio", "w_c_fe", "0.25, 0.5, 0.75")) or ())

    with der:
        st.subheader("Barreras y bandas")
        bar = sel("Barrera", "w_c_bar", BARS)
        L = U = None
        efecto, reb, mom, t1, t2, bgk = "ninguna", 0.0, "toque", 0.0, None, False
        if bar != "Sin barrera":
            c = st.columns(2)
            ef = radio("Al tocarla, el contrato", "w_c_efecto", ["Sale", "Entra"], cont=c[0],
                       help="Sale: se extingue. Entra: empieza a existir.")
            efecto = ef.lower()
            if bar in ("Inferior", "Banda (inferior y superior)"):
                L = num("Barrera inferior", "w_c_L", .8 * m.S0, c[1], min_value=.0001)
            if bar in ("Superior", "Banda (inferior y superior)"):
                U = num("Barrera superior", "w_c_U", 1.25 * m.S0, c[1], min_value=.0001)
            c = st.columns(2)
            reb = num("Rebate", "w_c_reb", 0.0, c[0], min_value=0.0,
                      help="Sale: se paga si la barrera se toca. Entra: se paga en T si nunca entró.")
            if efecto == "sale":
                mom = "toque" if radio("El rebate se paga", "w_c_rm", ["Al tocar", "En T"], cont=c[1]) == "Al tocar" else "vencimiento"
            c = st.columns(2)
            t1 = num("Vigilar desde", "w_c_t1", 0.0, c[0], min_value=0.0, max_value=float(T))
            t2 = num("Vigilar hasta", "w_c_t2", float(T), c[1], min_value=0.0, max_value=float(T))
            bgk = casilla("Aproximar vigilancia continua", "w_c_bgk", True,
                          help="Ajusta la barrera con la corrección de Broadie-Glasserman-Kou. Desmárcala si solo se observa en las fechas de la malla.")

        st.subheader("Otras cláusulas")
        rL = rU = None
        rc = 0.0
        if casilla("Cupón por días en banda", "w_c_rango"):
            c = st.columns(3)
            rL = num("Desde", "w_c_r_L", .9 * m.S0, c[0], min_value=0.0)
            rU = num("Hasta", "w_c_r_U", 1.1 * m.S0, c[1], min_value=0.0)
            rc = num("Cupón total", "w_c_r_c", 10.0, c[2], min_value=0.0)
        a_niv, a_cup, a_f = None, 0.0, ()
        if casilla("Autollamado", "w_c_auto"):
            c = st.columns(3)
            a_niv = num("Nivel", "w_c_a_niv", float(m.S0), c[0], min_value=0.0)
            a_cup = num("Cupón por fecha", "w_c_a_cup", 4.0, c[1], min_value=0.0)
            a_f = tuple(lista_num(txt("Fechas", "w_c_a_f", "0.25, 0.5, 0.75", cont=c[2])) or ())
            st.caption("Si S ≥ nivel en una fecha, paga nominal × monto más cupón × número de fecha, y termina.")

    return Contrato(base=base, referencia=ref, strike_tipo=st_tipo, K=K1, K2=K2, monto=monto, potencia=pot, t_prom=t_prom, prom_con_S0=con_S0,
                    t_adelantado=t_ad, alfa_adelantado=al_ad, participacion=part, tope=tope, piso=piso, nominal=nom,
                    barrera=efecto, L=L, U=U, t_bar_ini=t1, t_bar_fin=t2, bgk=bgk, rebate=reb, rebate_momento=mom,
                    autollamado=a_niv, autollamado_cupon=a_cup, autollamado_fechas=a_f, rango_L=rL, rango_U=rU,
                    rango_cupon=rc, ejercicio=ej.lower(), fechas_ej=fechas, expresion=expr)


def _validar(c: Contrato, T):
    if c.L is not None and c.U is not None and c.L >= c.U:
        return "La barrera inferior debe ser menor que la superior."
    if c.t_bar_fin is not None and c.barrera != "ninguna" and c.t_bar_ini >= c.t_bar_fin:
        return "La vigilancia debe empezar antes de terminar."
    if c.base in ("call_spread", "put_spread") and c.K2 <= c.K:
        return "En un spread, K₂ debe ser mayor que K."
    if c.ejercicio == "bermuda" and not c.fechas_ej:
        return "Escribe al menos una fecha de ejercicio, por ejemplo 0.5."
    if c.ejercicio != "europeo" and (c.autollamado is not None or c.rango_cupon):
        return "El ejercicio anticipado no se combina con autollamado ni con cupón por banda."
    if c.autollamado is not None and not c.autollamado_fechas:
        return "Escribe las fechas de autollamado."
    return None


def _resumen(c: Contrato):
    pz = [f"Pago: {[k for k, v in BASES.items() if v == c.base][0].lower()}"]
    if c.base not in ("monto", "expresion"):
        pz.append(f"medido sobre {[k for k, v in REFS.items() if v == c.referencia][0]}")
    if c.barrera != "ninguna":
        niveles = " y ".join(f"{x:,.2f}" for x in (c.L, c.U) if x is not None)
        pz.append(f"{'se extingue' if c.barrera == 'sale' else 'se activa'} al tocar {niveles}")
    if c.rebate:
        pz.append(f"rebate de {c.rebate:,.2f}")
    if c.tope is not None:
        pz.append(f"tope {c.tope:,.2f}")
    if c.piso is not None:
        pz.append(f"piso {c.piso:,.2f}")
    if c.rango_cupon:
        pz.append(f"cupón por banda {c.rango_cupon:,.2f}")
    if c.autollamado is not None:
        pz.append(f"autollamado en {c.autollamado:,.2f}")
    pz.append(f"ejercicio {c.ejercicio}")
    return ", ".join(pz) + "."


def pagina():
    m = mercado()
    K, T = K_T()
    encabezado("Constructor de contratos", "Combina pago, barreras, bandas, rebate, topes y ejercicio. "
               "Los campos del mercado se toman de la barra lateral.")
    c = st.columns([2, 1])
    pre = c[0].selectbox("Empezar desde un ejemplo", ["Contrato en blanco", "Doble sin toque de clase", "Lookback de clase",
                         "Asiática con barrera y rebate", "Put americano con barrera", "Call con tope y participación",
                         "Nota autollamable", "Acumulador en banda con salida"], key="c_preset")
    c[1].button("Cargar ejemplo", on_click=_preset, args=(pre,), width="stretch")

    con = _formulario(m, K, T)
    st.subheader("Valuación")
    c = st.columns(4)
    metodo = sel("Método", "w_c_met", ["Monte Carlo", "Árbol exacto (hasta 16 pasos)"], cont=c[0])
    if metodo == "Monte Carlo":
        pasos = num("Fechas de la malla", "w_c_pa", 126, c[1], min_value=1, max_value=504,
                    help="Son las fechas en que se observa S. Más fechas dan más precisión en barreras y promedios.")
        n = sel("Trayectorias", "w_c_n", [20_000, 50_000, 100_000], default=50_000, cont=c[2], format_func=lambda x: f"{x:,}")
    else:
        pasos = num("Pasos N", "w_c_pa_t", 10, c[1], min_value=1, max_value=16)
        n = 0
    griega = casilla("Calcular griegas", "w_c_gr", cont=c[3], help="Tarda unas diez veces más.")

    err = _validar(con, T)
    if err:
        aviso("Falta ajustar el contrato", err)
        return
    if metodo == "Monte Carlo" and n * pasos > 40_000_000:
        aviso("Esa simulación necesita demasiada memoria", "Reduce las trayectorias o las fechas de la malla: "
              "trayectorias por fechas debe quedar por debajo de 40 millones.")
        return
    if metodo == "Monte Carlo" and con.ejercicio == "americano" and pasos > 100:
        st.caption("Con ejercicio americano la malla se limita a 100 fechas.")
        pasos = 100
    if st.button("Valuar contrato", type="primary"):
        with st.spinner("Valuando"):
            try:
                if metodo == "Monte Carlo":
                    r = si.valuar_mc(con, m, T, int(pasos), int(n))
                    tt, S = si.simular(m, T, min(int(pasos), 200), 60)
                else:
                    r = si.valuar_arbol(con, m, T, int(pasos))
                    tt, S = si.simular(m, T, int(pasos), 60)
                g = si.griegas_mc(con, m, T, int(pasos), min(int(n) or 20_000, 50_000)) if griega and metodo == "Monte Carlo" else None
            except (ValueError, ZeroDivisionError, FloatingPointError) as e:
                aviso("No pude valuarlo", str(e))
                return
            r_vp = r.vp
            r.vp = None
            st.session_state["_c_res"] = dict(r=r, vp=r_vp, tt=tt, S=S, g=g, con=repr(con), m=m, T=T, res=_resumen(con), c=con)

    d = st.session_state.get("_c_res")
    if d is None:
        st.caption("Presiona Valuar contrato para ver el resultado.")
        return
    if d["con"] != repr(con) or d["m"] != m or d["T"] != T:
        aviso("Cambiaste algún dato", "El resultado que ves corresponde al contrato anterior. Presiona Valuar contrato.", "neutro")
    r = d["r"]
    st.markdown(f"**Tu contrato.** {d['res']}")
    tarjetas = [("Valor del contrato", r.precio, f"error estándar {r.ee:,.4f}" if r.ee else "suma exacta sobre el árbol", False),
                ("Intervalo al 95 %", f"{r.ic[0]:,.3f} a {r.ic[1]:,.3f}", "", True)]
    if r.p_toque is not None:
        tarjetas.append(("Probabilidad Q de tocar", f"{r.p_toque:.2%}", "", True))
    if r.p_autollamado is not None:
        tarjetas.append(("Probabilidad Q de autollamado", f"{r.p_autollamado:.2%}", "", True))
    if r.p_temprano is not None:
        tarjetas.append(("Probabilidad de ejercer antes", f"{r.p_temprano:.2%}", "el precio es una cota inferior", True))
    elif r.t_medio is not None and len(tarjetas) < 4:
        tarjetas.append(("Plazo medio del pago", f"{r.t_medio:.3f} años", "", True))
    c = st.columns(max(len(tarjetas), 3))
    for i, (et, va, pie, sec) in enumerate(tarjetas):
        caja(et, va, pie, sec, cont=c[i])
    if d["g"]:
        c = st.columns(4)
        for i, (k, v) in enumerate(d["g"].items()):
            caja(k, v, secundaria=True, cont=c[i])

    con_d = d["c"]
    izq, der = st.columns(2)
    with izq:
        fig = figura("Trayectorias de muestra", "Tiempo (años)", "S", 340)
        for i in range(min(40, d["S"].shape[0])):
            fig.add_trace(go.Scatter(x=d["tt"], y=d["S"][i], mode="lines", line=dict(color="rgba(42,98,214,.28)", width=1),
                                     hoverinfo="skip", showlegend=False))
        for nivel, et in ((con_d.L, "Barrera"), (con_d.U, "Barrera"), (con_d.autollamado, "Autollamado")):
            if nivel is not None:
                fig.add_hline(y=nivel, line=dict(color="#d1344b", width=2, dash="dash"), annotation_text=et,
                              annotation_font=dict(size=11))
        fig.update_layout(hovermode=False)
        graficar(fig)
    with der:
        if d["vp"] is not None and r.ee:
            fig = figura("Distribución del valor presente del pago", "Valor presente", "Trayectorias", 340)
            fig.add_histogram(x=d["vp"], nbinsx=50, marker_color="#2a62d6", showlegend=False)
            fig.update_layout(hovermode="closest", bargap=.04)
            graficar(fig)
