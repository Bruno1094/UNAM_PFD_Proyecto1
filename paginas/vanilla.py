"""Black-Scholes, árboles binomiales, griegas y volatilidad implícita."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import brentq

from .base import Mercado, N, n


# ---------------------------------------------------------------------------
# Black-Scholes sobre el forward prepagado
# ---------------------------------------------------------------------------
def d1d2(P, KB, sigma, T):
    raiz = sigma * np.sqrt(T)
    d1 = (np.log(P / KB) + 0.5 * raiz ** 2) / raiz
    return d1, d1 - raiz


def black(P, KB, sigma, T, tipo="call"):
    """c = P·N(d1) - K·B·N(d2).  Con P = S0 y B = e^{-rT}."""
    P = np.asarray(P, dtype=float)
    KB = np.asarray(KB, dtype=float)
    if T <= 0 or sigma <= 0:
        v = np.maximum(P - KB, 0.0) if tipo == "call" else np.maximum(KB - P, 0.0)
    else:
        P_seg = np.maximum(P, 1e-300)
        d1, d2 = d1d2(P_seg, KB, sigma, T)
        if tipo == "call":
            v = P_seg * N(d1) - KB * N(d2)
        else:
            v = KB * N(-d2) - P_seg * N(-d1)
    return float(v) if v.ndim == 0 else v


def europea(m: Mercado, K, T, tipo="call"):
    return black(m.prepago(T), K * m.B(T), m.sigma, T, tipo)


def detalle_bs(m: Mercado, K, T) -> dict:
    P, KB = m.prepago(T), K * m.B(T)
    d1, d2 = d1d2(P, KB, m.sigma, T)
    return {"P": P, "KB": KB, "B": m.B(T), "d1": d1, "d2": d2,
            "Nd1": N(d1), "Nd2": N(d2), "N-d1": N(-d1), "N-d2": N(-d2)}


# ---------------------------------------------------------------------------
# Caso Bernoulli (un periodo)
# ---------------------------------------------------------------------------
def bernoulli(S0, Su, Sd, Xu, Xd, B) -> dict:
    alfa = (Xu - Xd) / (Su - Sd)
    beta = B * (Xu - alfa * Su)
    q = (S0 / B - Sd) / (Su - Sd)
    V0 = B * (q * Xu + (1 - q) * Xd)
    return {"alfa": alfa, "beta": beta, "q": q, "V0": V0, "V0_replica": alfa * S0 + beta}


def no_recombinante(S, B, X) -> dict:
    """Árbol de 2 pasos no recombinante.

    S = (S1, S2, S3, S4, S5, S6, S7)
    B = (B(0,1δ), B(1δ,2δ) en el nodo 2, B(1δ,2δ) en el nodo 3)
    X = (X4, X5, X6, X7)
    """
    S1, S2, S3, S4, S5, S6, S7 = S
    b1, b2, b3 = B
    X4, X5, X6, X7 = X
    q1 = (S1 / b1 - S2) / (S3 - S2)
    q2 = (S2 / b2 - S4) / (S5 - S4)
    q3 = (S3 / b3 - S6) / (S7 - S6)
    V2 = b2 * (q2 * X5 + (1 - q2) * X4)
    V3 = b3 * (q3 * X7 + (1 - q3) * X6)
    return {"q1": q1, "q2": q2, "q3": q3, "V2": V2, "V3": V3, "V0": b1 * (q1 * V3 + (1 - q1) * V2),
            "alfa1": (V3 - V2) / (S3 - S2), "alfa2": (X5 - X4) / (S5 - S4), "alfa3": (X7 - X6) / (S7 - S6)}


# ---------------------------------------------------------------------------
# Árbol multiplicativo
# ---------------------------------------------------------------------------
def factores(m: Mercado, T, pasos, metodo="crr", mu=None, u=None, d=None):
    """Regresa (u, d, q) con q neutral al riesgo."""
    delta = T / pasos
    raiz = m.sigma * np.sqrt(delta)
    if metodo == "crr":
        u, d = np.exp(raiz), np.exp(-raiz)
    elif metodo == "jr":                     # media neutral al riesgo exacta
        nu = m.b - 0.5 * m.sigma ** 2
        u, d = np.exp(nu * delta + raiz), np.exp(nu * delta - raiz)
    elif metodo == "notas":                  # u = e^{μδ + σ√δ}, d = e^{μδ - σ√δ}, μ propia
        u, d = np.exp(mu * delta + raiz), np.exp(mu * delta - raiz)
    elif metodo != "manual":
        raise ValueError(metodo)
    q = (np.exp(m.b * delta) - d) / (u - d)
    return float(u), float(d), float(q)


def _pv_restante(m: Mercado, T, pasos):
    """Valor en cada paso de los pagos futuros, más el pago acumulado en cada paso."""
    delta = T / pasos
    tiempos = np.arange(pasos + 1) * delta
    resto = np.zeros(pasos + 1)
    if m.pagos:
        for ti, di in m.pagos_hasta(T):
            resto += np.where(tiempos < ti - 1e-12, di * np.exp(-m.b * (ti - tiempos)), 0.0)
    pago_paso = np.zeros(pasos)                # valor en t_{k+1} del pago ocurrido en (t_k, t_{k+1}]
    for ti, di in m.pagos_hasta(T):
        k = min(max(int(np.ceil(ti / delta - 1e-12)) - 1, 0), pasos - 1)
        pago_paso[k] += di * np.exp(m.b * (tiempos[k + 1] - ti))
    return tiempos, resto, pago_paso


def pasos_de_fechas(fechas, T, pasos):
    delta = T / pasos
    return sorted({min(max(int(round(t / delta)), 0), pasos) for t in fechas if 0 <= t <= T})


@dataclass
class ResultadoArbol:
    precio: float
    u: float
    d: float
    q: float
    B: float
    S: list = field(default_factory=list)      # S[k][j]
    V: list = field(default_factory=list)      # V[k][j]
    ejercido: list = field(default_factory=list)
    alfa: list = field(default_factory=list)
    beta: list = field(default_factory=list)
    continuacion: list = field(default_factory=list)
    frontera: list = field(default_factory=list)  # (t, S crítico)


def arbol(m: Mercado, T, pasos, payoff, estilo="europea", fechas=(), metodo="crr",
          mu=None, u=None, d=None, guardar=False) -> ResultadoArbol:
    """h_N(y) = X(y);  h_k(y) = B[q·h_{k+1}(uy) + (1-q)·h_{k+1}(dy)]  y, si se puede
    ejercer en k, h_k(y) = max{X(y), continuación}.

    Con ingresos fijos el árbol se construye sobre S* = S0 - C y a cada nodo se le
    suma el valor presente de los pagos que faltan, así sigue recombinando.
    """
    u, d, q = factores(m, T, pasos, metodo, mu, u, d)
    delta = T / pasos
    B = float(m.B(delta))
    tiempos, resto, pago_paso = _pv_restante(m, T, pasos)
    S_est = m.S_estrella(T)
    lu, ld = np.log(u), np.log(d)

    def nodos(k):
        j = np.arange(k + 1)
        return np.maximum(S_est, 1e-300) * np.exp(j * lu + (k - j) * ld) + resto[k]

    if estilo == "americana":
        ejercer = set(range(0, pasos))
    elif estilo == "bermuda":
        ejercer = set(pasos_de_fechas(fechas, T, pasos))
    else:
        ejercer = set()

    S_fin = nodos(pasos)
    V = np.asarray(payoff(S_fin), dtype=float).copy()
    r = ResultadoArbol(0.0, u, d, q, B)
    if guardar:
        r.S = [None] * (pasos + 1); r.V = [None] * (pasos + 1)
        r.ejercido = [None] * (pasos + 1); r.alfa = [None] * pasos
        r.beta = [None] * pasos; r.continuacion = [None] * pasos
        r.S[pasos], r.V[pasos] = S_fin, V.copy()
        r.ejercido[pasos] = np.zeros(pasos + 1, bool)
    bE = float(m.BE(delta))
    for k in range(pasos - 1, -1, -1):
        Vu, Vd = V[1:], V[:-1]
        cont = B * (q * Vu + (1 - q) * Vd)
        Sk = nodos(k) if (k in ejercer or guardar) else None
        ej = np.zeros(k + 1, bool)
        if k in ejercer:
            X = np.asarray(payoff(Sk), dtype=float)
            ej = X > cont + 1e-12
            nuevo = np.where(ej, X, cont)
            if ej.any() and not ej.all():
                lim = Sk[ej]
                r.frontera.append((tiempos[k], float(lim.max() if lim.max() < Sk[~ej].min() else lim.min())))
        else:
            nuevo = cont
        if guardar:
            S1 = r.S[k + 1]
            Delta = (Vu - Vd) / (S1[1:] - S1[:-1])
            r.alfa[k] = bE * Delta
            r.beta[k] = B * (Vu - Delta * (S1[1:] + pago_paso[k]))
            r.S[k], r.V[k], r.ejercido[k], r.continuacion[k] = Sk, nuevo.copy(), ej, cont
        V = nuevo
    r.precio = float(V[0])
    r.frontera.sort()
    return r


def vanilla_arbol(m, K, T, tipo="call", estilo="europea", pasos=500, fechas=(), metodo="crr",
                  mu=None, u=None, d=None, guardar=False) -> ResultadoArbol:
    if tipo == "call":
        pf = lambda S: np.maximum(S - K, 0.0)
    else:
        pf = lambda S: np.maximum(K - S, 0.0)
    return arbol(m, T, pasos, pf, estilo, fechas, metodo, mu, u, d, guardar)


def precio_estilo(m, K, T, tipo, estilo, pasos=500, fechas=()):
    """Precio con variable de control: el error del árbol europeo se le resta al estilo."""
    if estilo == "europea":
        return float(europea(m, K, T, tipo))
    v = vanilla_arbol(m, K, T, tipo, estilo, pasos, fechas).precio
    ve = vanilla_arbol(m, K, T, tipo, "europea", pasos).precio
    return max(v + float(europea(m, K, T, tipo)) - ve,
               float(np.maximum(m.S0 - K, 0) if tipo == "call" else np.maximum(K - m.S0, 0))
               if estilo == "americana" else 0.0)


def enumerar_trayectorias(m: Mercado, T, pasos, metodo="crr", mu=None, u=None, d=None):
    """Todas las trayectorias de un árbol de N pasos, con su probabilidad bajo Q."""
    if pasos > 18:
        raise ValueError("Con más de 18 pasos hay demasiadas trayectorias; usa Monte Carlo.")
    u, d, q = factores(m, T, pasos, metodo, mu, u, d)
    tiempos, resto, _ = _pv_restante(m, T, pasos)
    bits = ((np.arange(2 ** pasos)[:, None] >> np.arange(pasos - 1, -1, -1)) & 1).astype(float)
    subidas = np.hstack([np.zeros((bits.shape[0], 1)), np.cumsum(bits, axis=1)])
    k = np.arange(pasos + 1)
    S = m.S_estrella(T) * u ** subidas * d ** (k - subidas) + resto
    pesos = q ** subidas[:, -1] * (1 - q) ** (pasos - subidas[:, -1])
    return tiempos, S, pesos


# ---------------------------------------------------------------------------
# Griegas por diferencias centrales (sirven para cualquier valuador)
# ---------------------------------------------------------------------------
def griegas(valuar, m: Mercado, T, h_S=None) -> dict:
    """valuar(m, T) regresa un precio. Se usan diferencias centrales."""
    h = h_S or 0.01 * float(m.S0)
    v = valuar(m, T)
    vu, vd = valuar(m.con(S0=m.S0 + h), T), valuar(m.con(S0=m.S0 - h), T)
    hs = min(0.01, m.sigma / 2)
    vega = (valuar(m.con(sigma=m.sigma + hs), T) - valuar(m.con(sigma=m.sigma - hs), T)) / (2 * hs)
    hr = 1e-4
    rho = (valuar(m.con(r=m.r + hr), T) - valuar(m.con(r=m.r - hr), T)) / (2 * hr)
    hq = 1e-4
    psi = (valuar(m.con(q=m.q + hq), T) - valuar(m.con(q=m.q - hq), T)) / (2 * hq)
    ht = min(1 / 365, T / 4)
    theta = (valuar(m.avanzar(ht), T - ht) - v) / ht
    return {"Precio": v, "Delta": (vu - vd) / (2 * h), "Gamma": (vu - 2 * v + vd) / h ** 2,
            "Vega": vega, "Theta": theta, "Rho": rho, "Psi": psi}


def griegas_analiticas(m: Mercado, K, T, tipo="call") -> dict:
    """Griegas europeas cerradas; se usan para gráficas y como verificación."""
    P, KB = m.prepago(T), K * m.B(T)
    d1, d2 = d1d2(P, KB, m.sigma, T)
    be = m.BE(T)
    raiz = np.sqrt(T)
    signo = 1 if tipo == "call" else -1
    delta = be * (N(d1) if tipo == "call" else N(d1) - 1)
    gamma = be ** 2 * n(d1) / (P * m.sigma * raiz)
    vega = P * n(d1) * raiz
    # Sin pagos discretos, la fórmula clásica; con ellos se usa el motor numérico.
    theta = (-P * n(d1) * m.sigma / (2 * raiz)
             - signo * m.rc * KB * N(signo * d2) + signo * m.qc * P * N(signo * d1))
    rho_c = signo * T * KB * N(signo * d2)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho_c}


# ---------------------------------------------------------------------------
# Volatilidad implícita
# ---------------------------------------------------------------------------
def vol_implicita(precio, m: Mercado, K, T, tipo="call", estilo="europea", pasos=200, fechas=()):
    """Regresa None si el precio queda fuera de lo alcanzable con cualquier volatilidad."""
    def f(s):
        mm = m.con(sigma=s)
        if estilo == "europea":
            return float(europea(mm, K, T, tipo)) - precio
        return vanilla_arbol(mm, K, T, tipo, estilo, pasos, fechas).precio - precio
    lo, hi = 1e-4, 5.0
    try:
        if f(lo) * f(hi) > 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-7))
    except ValueError:
        return None
