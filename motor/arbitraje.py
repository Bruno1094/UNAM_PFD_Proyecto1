"""Detección de arbitraje y estrategia con todos sus flujos.

`pura`  : la ganancia no depende de ningún modelo.
`modelo`: depende de que el modelo (la volatilidad) sea correcto, así que no es arbitraje.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .base import Mercado


@dataclass
class Estrategia:
    titulo: str
    existe: bool
    tipo: str = "ninguna"        # ninguna | pura | modelo
    ganancia_hoy: float = 0.0
    ganancia_T: float = 0.0
    desviacion: float = 0.0
    piernas: list = field(default_factory=list)
    nota: str = ""


def _p(op, hoy, t_sube, t_baja):
    return {"Operación": op, "Flujo hoy": hoy, "En T si S_T > K": t_sube, "En T si S_T ≤ K": t_baja}


def _piernas_subyacente(m: Mercado, T, compra: bool, precio_S):
    """Pierna de acciones, con el ingreso comprometido a pagar el préstamo correspondiente."""
    be = float(m.BE(T))
    c = m.vp_ingresos(T)
    unidad = f"{be:.4f} títulos" if m.qc else "1 título"
    if compra:
        pp = [_p(f"Comprar {unidad}" + (", reinvirtiendo el rendimiento" if m.qc else ""), -precio_S * be, "+S_T", "+S_T")]
        if c:
            pp.append(_p("Pedir prestado el VP de los ingresos; se paga con ellos", c * be, "0", "0"))
    else:
        pp = [_p(f"Vender en corto {unidad}", precio_S * be, "-S_T", "-S_T")]
        if c:
            pp.append(_p("Invertir el VP de los ingresos; cubren lo que se debe al prestamista", -c * be, "0", "0"))
    return pp


# ---------------------------------------------------------------------------
# Paridad put-call
# ---------------------------------------------------------------------------
def paridad_europea(m: Mercado, K, T, c_bid, c_ask, p_bid, p_ask, s_bid=None, s_ask=None, comision=0.0):
    """c - p = S0·B_E - C - K·B."""
    s_bid = m.S0 if s_bid is None else s_bid
    s_ask = m.S0 if s_ask is None else s_ask
    B, be, vp = float(m.B(T)), float(m.BE(T)), m.vp_ingresos(T)
    costo = 3 * comision
    conversion = c_bid - p_ask - (s_ask - vp) * be + K * B - costo
    reversa = -c_ask + p_bid + (s_bid - vp) * be - K * B - costo
    desv = (0.5 * (c_bid + c_ask) - 0.5 * (p_bid + p_ask)) - ((m.S0 - vp) * be - K * B)
    if conversion > 1e-10:
        pp = [_p("Vender call", c_bid, "-(S_T - K)", "0"), _p("Comprar put", -p_ask, "0", "+(K - S_T)")]
        pp += _piernas_subyacente(m, T, True, s_ask)
        pp += [_p("Pedir prestado K·B(0,T)", K * B, "-K", "-K")]
        if costo:
            pp.append(_p("Comisiones", -costo, "0", "0"))
        return Estrategia("Conversión: el call está caro frente al put", True, "pura", conversion,
                          conversion / B, desv, pp,
                          "Al vencimiento la posición se cierra sola, gane quien gane: entregas el título a cambio de K, "
                          "y K paga el préstamo. La ganancia queda asegurada hoy.")
    if reversa > 1e-10:
        pp = [_p("Comprar call", -c_ask, "+(S_T - K)", "0"), _p("Vender put", p_bid, "0", "-(K - S_T)")]
        pp += _piernas_subyacente(m, T, False, s_bid)
        pp += [_p("Invertir K·B(0,T)", -K * B, "+K", "+K")]
        if costo:
            pp.append(_p("Comisiones", -costo, "0", "0"))
        return Estrategia("Reversión: el call está barato frente al put", True, "pura", reversa,
                          reversa / B, desv, pp,
                          "Al vencimiento compras el título en K con lo invertido y cierras el corto. "
                          "La ganancia queda asegurada hoy.")
    return Estrategia("La paridad se cumple dentro de los costos", False, desviacion=desv,
                      nota="Ni la conversión ni la reversión dejan ganancia después de comprar al ask, "
                           "vender al bid y pagar comisiones.")


def paridad_americana(m: Mercado, K, T, c_bid, c_ask, p_bid, p_ask, s_bid=None, s_ask=None, comision=0.0):
    """Cotas: S0·B_E - C - K <= c - p <= S0 - K·B."""
    s_bid = m.S0 if s_bid is None else s_bid
    s_ask = m.S0 if s_ask is None else s_ask
    B, be, vp = float(m.B(T)), float(m.BE(T)), m.vp_ingresos(T)
    costo = 3 * comision
    gan_sup = c_bid - p_ask - s_ask + K * B - costo
    gan_inf = -c_ask + p_bid + (s_bid - vp) * be - K - costo
    inf, sup = (m.S0 - vp) * be - K, m.S0 - K * B
    mid = 0.5 * (c_bid + c_ask) - 0.5 * (p_bid + p_ask)
    desv = mid - sup if mid > sup else (mid - inf if mid < inf else 0.0)
    if gan_sup > 1e-10:
        pp = [_p("Vender call americano", c_bid, "-(S_T - K)", "0"), _p("Comprar put americano", -p_ask, "0", "+(K - S_T)"),
              _p("Comprar 1 título", -s_ask, "+S_T", "+S_T"), _p("Pedir prestado K·B(0,T)", K * B, "-K", "-K")]
        if costo:
            pp.append(_p("Comisiones", -costo, "0", "0"))
        return Estrategia("c - p rompe la cota superior", True, "pura", gan_sup, gan_sup / B, desv, pp,
                          "Si te ejercen antes, entregas el título y recibes K, que alcanza para pagar el préstamo "
                          "y además te quedas con el put. Los ingresos del título son ganancia adicional.")
    if gan_inf > 1e-10:
        pp = [_p("Comprar call americano", -c_ask, "+(S_T - K)", "0"), _p("Vender put americano", p_bid, "0", "-(K - S_T)"),
              _p("Vender en corto el título", s_bid * be - (vp * be), "-S_T", "-S_T"),
              _p("Invertir K a la vista", -K, f"+K/B = {K / B:,.4f}", f"+K/B = {K / B:,.4f}")]
        if costo:
            pp.append(_p("Comisiones", -costo, "0", "0"))
        return Estrategia("c - p rompe la cota inferior", True, "pura", gan_inf, gan_inf / B, desv, pp,
                          "Si te ejercen el put antes, compras el título en K con lo invertido, cierras el corto y "
                          "conservas el call. El dinero se invierte a la vista para tenerlo disponible.")
    return Estrategia("c - p está dentro de las cotas americanas", False, desviacion=desv,
                      nota=f"Las cotas son {inf:,.4f} ≤ c - p ≤ {sup:,.4f}.")


# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------
def forward_mercado(m: Mercado, T, F_bid, F_ask, s_bid=None, s_ask=None, comision=0.0):
    s_bid = m.S0 if s_bid is None else s_bid
    s_ask = m.S0 if s_ask is None else s_ask
    B, be, vp = float(m.B(T)), float(m.BE(T)), m.vp_ingresos(T)
    costo = 2 * comision / B
    teo = m.forward(T)
    F_compra = (s_ask - vp) * be / B
    F_venta = (s_bid - vp) * be / B
    desv = 0.5 * (F_bid + F_ask) - teo
    if F_bid - F_compra - costo > 1e-10:
        g = F_bid - F_compra - costo
        pp = _piernas_subyacente(m, T, True, s_ask)
        pp += [_p("Pedir prestado lo necesario", (s_ask - vp) * be, f"-{(s_ask - vp) * be / B:,.4f}", f"-{(s_ask - vp) * be / B:,.4f}"),
               _p(f"Vender forward a {F_bid:,.4f}", 0.0, f"+{F_bid:,.4f} - S_T", f"+{F_bid:,.4f} - S_T")]
        if comision:
            pp.append(_p("Comisiones", -2 * comision, "0", "0"))
        return Estrategia("Cash and carry: el forward está caro", True, "pura", g * B, g, desv, pp,
                          "Compras con dinero prestado y fijas hoy el precio de venta. El préstamo se paga con la venta.")
    if F_venta - F_ask - costo > 1e-10:
        g = F_venta - F_ask - costo
        pp = _piernas_subyacente(m, T, False, s_bid)
        pp += [_p("Invertir lo recibido", -(s_bid - vp) * be, f"+{(s_bid - vp) * be / B:,.4f}", f"+{(s_bid - vp) * be / B:,.4f}"),
               _p(f"Comprar forward a {F_ask:,.4f}", 0.0, f"+S_T - {F_ask:,.4f}", f"+S_T - {F_ask:,.4f}")]
        if comision:
            pp.append(_p("Comisiones", -2 * comision, "0", "0"))
        return Estrategia("Cash and carry inverso: el forward está barato", True, "pura", g * B, g, desv, pp,
                          "Vendes en corto, inviertes y fijas hoy el precio de recompra que queda "
                          "cubierto con lo resultante de la inversión.")
    return Estrategia("El forward está en línea", False, desviacion=desv,
                      nota=f"Sin arbitraje mientras {F_venta - costo:,.4f} ≤ F ≤ {F_compra + costo:,.4f}, una vez incluidos costos.")


# ---------------------------------------------------------------------------
# Cotas libres de modelo
# ---------------------------------------------------------------------------
def cotas(m: Mercado, K, T, tipo, estilo):
    """Regresa (inferior, superior) libres de modelo."""
    P, KB = m.prepago(T), K * float(m.B(T))
    if tipo == "call":
        inf = max(P - KB, 0.0)
        sup = P if estilo == "europea" else m.S0
        if estilo == "americana":
            inf = max(inf, m.S0 - K)
    else:
        inf = max(KB - P, 0.0)
        sup = KB if estilo == "europea" else K
        if estilo == "americana":
            inf = max(inf, K - m.S0)
    return float(inf), float(sup)


def revisar_cotas(m: Mercado, K, T, tipo, estilo, precio) -> Estrategia:
    inf, sup = cotas(m, K, T, tipo, estilo)
    B = float(m.B(T))
    P, KB = m.prepago(T), K * B
    if precio > sup + 1e-10:
        g = precio - sup
        if tipo == "call":
            pp = [_p("Vender call", precio, "-(S_T - K)", "0"),
                  _p("Comprar el título", -sup, "+S_T", "+S_T")]
            return Estrategia("El call vale más que el subyacente", True, "pura", g, g / B, g, pp,
                              "En T te queda mín(S_T, K) ≥ 0, además de la ganancia de hoy.")
        pp = [_p("Vender put", precio, "0", "-(K - S_T)"), _p("Invertir el valor máximo a pagar", -sup, "+K", "+K")]
        return Estrategia("El put vale más que su pago máximo", True, "pura", g, g / B, g, pp,
                          "En T te queda mín(S_T, K) ≥ 0, además de la ganancia de hoy.")
    if precio < inf - 1e-10:
        g = inf - precio
        intrinseco = estilo == "americana" and inf == (m.S0 - K if tipo == "call" else K - m.S0)
        if tipo == "call":
            if intrinseco:
                pp = [_p("Comprar call", -precio, "0", "0"), _p("Ejercer de inmediato: pagar K", -K, "0", "0"),
                      _p("Vender el título recibido", m.S0, "0", "0")]
                return Estrategia("El call americano vale menos que su valor intrínseco", True, "pura", g, g / B, -g, pp,
                                  "Se ejecuta de inmediato, sin esperar al vencimiento.")
            pp = [_p("Comprar call", -precio, "+(S_T - K)", "0"), _p("Vender en corto el subyacente", P, "-S_T", "-S_T"),
                  _p("Invertir K·B(0,T)", -KB, "+K", "+K")]
            return Estrategia("El call vale menos que su cota inferior", True, "pura", g, g / B, -g, pp,
                              "En T te queda 0 si S_T > K, o K - S_T ≥ 0 en otro caso.")
        if intrinseco:
            pp = [_p("Comprar put", -precio, "0", "0"), _p("Comprar el título", -m.S0, "0", "0"),
                  _p("Ejercer de inmediato: recibir K", K, "0", "0")]
            return Estrategia("El put americano vale menos que su valor intrínseco", True, "pura", g, g / B, -g, pp,
                              "Se ejecuta de inmediato, sin esperar al vencimiento.")
        pp = [_p("Comprar put", -precio, "0", "+(K - S_T)"), _p("Comprar el subyacente", -P, "+S_T", "+S_T"),
              _p("Pedir prestado K·B(0,T)", KB, "-K", "-K")]
        return Estrategia("El put vale menos que su cota inferior", True, "pura", g, g / B, -g, pp,
                          "En T te queda S_T - K > 0 si S_T > K, o 0 en otro caso.")
    return Estrategia("El precio respeta las cotas libres de modelo", False,
                      nota=f"Rango sin arbitraje: {inf:,.4f} ≤ precio ≤ {sup:,.4f}.")


def contra_modelo(m: Mercado, K, T, tipo, precio_mkt, precio_teo, delta) -> Estrategia:
    """Diferencia entre mercado y modelo; se cubre en delta pero no es arbitraje."""
    dif = precio_mkt - precio_teo
    if abs(dif) < 1e-10:
        return Estrategia("El mercado coincide con el modelo", False)
    B = float(m.B(T))
    nombre = "call" if tipo == "call" else "put"
    if dif > 0:
        banco = precio_mkt - delta * m.S0
        pp = [_p(f"Vender {nombre}", precio_mkt, "según ejercicio", "según ejercicio"),
              _p(f"{'Comprar' if delta > 0 else 'Vender en corto'} {abs(delta):.4f} títulos (Δ)", -delta * m.S0, "se rebalancea", "se rebalancea"),
              _p("Invertir o financiar el saldo", -banco, "saldo capitalizado", "saldo capitalizado")]
        titulo = "La opción está cara frente al modelo: vende y cubre en delta"
    else:
        banco = -precio_mkt + delta * m.S0
        pp = [_p(f"Comprar {nombre}", -precio_mkt, "según ejercicio", "según ejercicio"),
              _p(f"{'Vender en corto' if delta > 0 else 'Comprar'} {abs(delta):.4f} títulos (Δ)", delta * m.S0, "se rebalancea", "se rebalancea"),
              _p("Invertir o financiar el saldo", -banco, "saldo capitalizado", "saldo capitalizado")]
        titulo = "La opción está barata frente al modelo: compra y cubre en delta"
    return Estrategia(titulo, True, "modelo", abs(dif), abs(dif) / B, dif, pp,
                      "Esto no es arbitraje. La ganancia esperada solo se materializa si la volatilidad realizada coincide "
                      "con la del modelo y si rebalanceas Δ con frecuencia; hay riesgo de volatilidad, de brincos y costos.")


def binomial_mercado(S0, Su, Sd, B, Xu, Xd, V_mkt) -> Estrategia:
    """En un mercado completo de un periodo la diferencia sí es arbitraje puro."""
    rf = S0 / B
    if rf >= Su:
        g = S0 - B * Su
        pp = [{"Operación": "Vender en corto 1 título", "Flujo hoy": S0, "Si sube": -Su, "Si baja": -Sd},
              {"Operación": f"Invertir {B * Su:,.4f}", "Flujo hoy": -B * Su, "Si sube": Su, "Si baja": Su}]
        return Estrategia("El bono le gana al título en todos los escenarios", True, "pura", g, g / B, rf - Su, pp,
                          "q > 1: no existe medida neutral al riesgo.")
    if rf <= Sd:
        g = B * Sd - S0
        pp = [{"Operación": "Comprar 1 título", "Flujo hoy": -S0, "Si sube": Su, "Si baja": Sd},
              {"Operación": f"Pedir prestado {B * Sd:,.4f}", "Flujo hoy": B * Sd, "Si sube": -Sd, "Si baja": -Sd}]
        return Estrategia("El título le gana al bono en todos los escenarios", True, "pura", g, g / B, rf - Sd, pp,
                          "q < 0: no existe medida neutral al riesgo.")
    alfa = (Xu - Xd) / (Su - Sd)
    beta = B * (Xu - alfa * Su)
    teo = alfa * S0 + beta
    d = V_mkt - teo
    if abs(d) < 1e-10:
        return Estrategia("El derivado está bien valuado", False)
    s = 1 if d > 0 else -1
    pp = [{"Operación": "Vender el derivado" if s > 0 else "Comprar el derivado", "Flujo hoy": s * V_mkt, "Si sube": -s * Xu, "Si baja": -s * Xd},
          {"Operación": f"{'Comprar' if s * alfa > 0 else 'Vender en corto'} {abs(alfa):.4f} títulos (α)", "Flujo hoy": -s * alfa * S0, "Si sube": s * alfa * Su, "Si baja": s * alfa * Sd},
          {"Operación": f"{'Invertir' if s * beta > 0 else 'Pedir prestado'} {abs(beta):,.4f} (β)", "Flujo hoy": -s * beta, "Si sube": s * beta / B, "Si baja": s * beta / B}]
    return Estrategia("El derivado está caro frente a su réplica" if s > 0 else "El derivado está barato frente a su réplica",
                      True, "pura", abs(d), abs(d) / B, d, pp,
                      "El portafolio (α, β) replica el pago en ambos escenarios, por eso los flujos finales se cancelan.")


def arbitraje_arbol(v, V_mkt) -> Estrategia:
    """Árbol completo: se vende (o compra) el derivado y se sigue la cobertura dinámica."""
    d = V_mkt - v.precio
    if abs(d) < 1e-10:
        return Estrategia("El mercado coincide con el valor del árbol", False)
    s = 1 if d > 0 else -1
    a0, b0, S0, B = v.alfa[1], v.beta[1], v.a.S[1], v.a.B[1]
    pp = [{"Operación": "Vender el derivado" if s > 0 else "Comprar el derivado", "Flujo hoy": s * V_mkt},
          {"Operación": f"{'Comprar' if s * a0 > 0 else 'Vender en corto'} {abs(a0):.4f} títulos (α)", "Flujo hoy": -s * a0 * S0},
          {"Operación": f"{'Invertir' if s * b0 > 0 else 'Pedir prestado'} {abs(b0):,.4f} (β)", "Flujo hoy": -s * b0}]
    return Estrategia("El derivado está caro frente a su cobertura" if s > 0 else "El derivado está barato frente a su cobertura",
                      True, "pura", abs(d), abs(d) / float(np.prod([v.a.B[1]] * v.a.N)), d, pp,
                      "En cada nodo se rebalancea a la α y β de ese nodo sin poner ni sacar dinero, porque la estrategia se "
                      "autofinancia. Al vencimiento la cartera iguala el pago en todas las trayectorias, así que la ganancia de hoy queda segura.")


# ---------------------------------------------------------------------------
# Cadena de strikes
# ---------------------------------------------------------------------------
def cadena(m: Mercado, T, ks, calls, puts, minimo=1e-6) -> list:
    """Monotonía, pendiente, convexidad, paridad y box entre strikes contiguos.

    `minimo` descarta señales menores que el redondeo o los costos de operar.
    """
    orden = np.argsort(ks)
    ks = np.asarray(ks, float)[orden]
    calls = np.asarray(calls, float)[orden]
    puts = np.asarray(puts, float)[orden]
    B = float(m.B(T))
    P = m.prepago(T)
    hall = []

    def _h(titulo, g, piernas, nota):
        if g >= minimo:
            hall.append(Estrategia(titulo, True, "pura", g, g / B, g, piernas, nota))

    def f(op, hoy):
        return {"Operación": op, "Flujo hoy": hoy}

    for i, (k1, c1, p1) in enumerate(zip(ks, calls, puts)):
        d = (c1 - p1) - (P - k1 * B)
        if not np.isnan(d) and abs(d) > 1e-6:
            if d > 0:
                _h(f"Paridad en K = {k1:g}: call caro", d,
                   [f(f"Vender call {k1:g}", c1), f(f"Comprar put {k1:g}", -p1), f("Comprar el subyacente (prepago)", -P), f(f"Pedir prestado {k1:g}·B", k1 * B)],
                   "Conversión: en T entregas el título, recibes K y pagas el préstamo.")
            else:
                _h(f"Paridad en K = {k1:g}: call barato", -d,
                   [f(f"Comprar call {k1:g}", -c1), f(f"Vender put {k1:g}", p1), f("Vender en corto el subyacente (prepago)", P), f(f"Invertir {k1:g}·B", -k1 * B)],
                   "Reversión: en T compras el título en K con lo invertido y cierras el corto.")

    for nombre, v, signo in (("call", calls, 1), ("put", puts, -1)):
        for i in range(len(ks) - 1):
            k1, k2, v1, v2 = ks[i], ks[i + 1], v[i], v[i + 1]
            if np.isnan(v1) or np.isnan(v2):
                continue
            dif = signo * (v1 - v2)
            if dif < -1e-9:
                alto, bajo = (k1, k2) if signo > 0 else (k2, k1)
                _h(f"Monotonía en {nombre}s entre {k1:g} y {k2:g}", -dif,
                   [f(f"Comprar {nombre} {alto:g}", -(v1 if signo > 0 else v2)), f(f"Vender {nombre} {bajo:g}", (v2 if signo > 0 else v1))],
                   "Compras el derecho mejor por menos dinero; el pago final nunca es negativo.")
            maxi = (k2 - k1) * B
            if dif > maxi + 1e-9:
                malo, bueno = (k1, k2) if signo > 0 else (k2, k1)
                _h(f"Pendiente en {nombre}s entre {k1:g} y {k2:g}", dif - maxi,
                   [f(f"Vender {nombre} {malo:g}", (v1 if signo > 0 else v2)), f(f"Comprar {nombre} {bueno:g}", -(v2 if signo > 0 else v1)), f(f"Invertir {maxi:,.4f}", -maxi)],
                   "La diferencia de primas supera el valor presente de lo más que puedes pagar en T.")
        for i in range(len(ks) - 2):
            k1, k2, k3 = ks[i:i + 3]
            v1, v2, v3 = v[i:i + 3]
            if np.isnan([v1, v2, v3]).any():
                continue
            lam = (k3 - k2) / (k3 - k1)
            g = v2 - lam * v1 - (1 - lam) * v3
            if g > 1e-9:
                _h(f"Convexidad en {nombre}s: mariposa {k1:g}/{k2:g}/{k3:g}", g,
                   [f(f"Comprar {lam:.4f} {nombre} {k1:g}", -lam * v1), f(f"Vender 1 {nombre} {k2:g}", v2), f(f"Comprar {1 - lam:.4f} {nombre} {k3:g}", -(1 - lam) * v3)],
                   "Una mariposa nunca paga negativo en T, pero aquí te pagan por entrar.")

    for i in range(len(ks) - 1):
        k1, k2 = ks[i], ks[i + 1]
        costo = calls[i] - calls[i + 1] - puts[i] + puts[i + 1]
        justo = (k2 - k1) * B
        if np.isnan(costo):
            continue
        if costo < justo - 1e-6:
            _h(f"Box {k1:g}/{k2:g} barato", justo - costo,
               [f(f"Comprar call {k1:g}", -calls[i]), f(f"Vender call {k2:g}", calls[i + 1]), f(f"Vender put {k1:g}", puts[i]), f(f"Comprar put {k2:g}", -puts[i + 1])],
               f"El box paga {k2 - k1:g} con certeza en T.")
        elif costo > justo + 1e-6:
            _h(f"Box {k1:g}/{k2:g} caro", costo - justo,
               [f(f"Vender call {k1:g}", calls[i]), f(f"Comprar call {k2:g}", -calls[i + 1]), f(f"Comprar put {k1:g}", -puts[i]), f(f"Vender put {k2:g}", puts[i + 1])],
               f"Recibes hoy más que el valor presente de los {k2 - k1:g} que pagarás en T.")
    return sorted(hall, key=lambda e: -e.ganancia_hoy)
