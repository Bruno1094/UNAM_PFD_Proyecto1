"""Derivados exóticos con fórmula cerrada o integral de una dimensión.

Todas las funciones reciben Mercado. Las fórmulas de barrera, lookback y las demás
dependientes de la trayectoria suponen monitoreo continuo y sin pagos discretos.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .base import Mercado, N, n
from .vanilla import black, europea


def _datos(m: Mercado, T):
    return m.S_estrella(T), m.rc, m.qc, m.b, m.sigma


# ---------------------------------------------------------------------------
# Digitales y familia
# ---------------------------------------------------------------------------
def _dd(m, K, T):
    S, r, q, b, s = _datos(m, T)
    d1 = (np.log(S / K) + (b + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    return d1, d1 - s * np.sqrt(T)


def digital_efectivo(m, K, T, tipo="call", monto=1.0):
    _, d2 = _dd(m, K, T)
    return monto * m.B(T) * N(d2 if tipo == "call" else -d2)


def digital_activo(m, K, T, tipo="call"):
    d1, _ = _dd(m, K, T)
    return m.prepago(T) * N(d1 if tipo == "call" else -d1)


def gap(m, K_disparo, K_pago, T, tipo="call"):
    d1, d2 = _dd(m, K_disparo, T)
    if tipo == "call":
        return m.prepago(T) * N(d1) - K_pago * m.B(T) * N(d2)
    return K_pago * m.B(T) * N(-d2) - m.prepago(T) * N(-d1)


def supershare(m, K1, K2, T):
    """Paga S_T / K1 si K1 < S_T < K2."""
    return (digital_activo(m, K1, T) - digital_activo(m, K2, T)) / K1


def potencia(m, K, T, a, tipo="call"):
    """Payoff (S_T^a - K)+."""
    S, r, q, b, s = _datos(m, T)
    media = a * (np.log(S) + (b - 0.5 * s ** 2) * T)
    v = abs(a) * s * np.sqrt(T)
    F = np.exp(media + 0.5 * v ** 2)
    return _black_lognormal(F, K, v, m.B(T), tipo)


def _black_lognormal(F, K, v, B, tipo):
    d1 = (np.log(F / K) + 0.5 * v ** 2) / v
    d2 = d1 - v
    if tipo == "call":
        return B * (F * N(d1) - K * N(d2))
    return B * (K * N(-d2) - F * N(-d1))


# ---------------------------------------------------------------------------
# Barreras de una sola banda (Reiner-Rubinstein)
# ---------------------------------------------------------------------------
def barrera(m: Mercado, K, H, T, tipo="call", direccion="abajo", accion="sale", rebate=0.0):
    """direccion: abajo|arriba; accion: sale|entra.

    Rebate: en las de salida se paga al tocar; en las de entrada, en T si nunca entró.
    """
    S, r, q, b, s = _datos(m, T)
    raiz = s * np.sqrt(T)
    vanilla = float(europea(m, K, T, tipo))
    tocada = S <= H if direccion == "abajo" else S >= H
    if tocada:
        return vanilla if accion == "entra" else rebate

    phi = 1 if tipo == "call" else -1
    eta = 1 if direccion == "abajo" else -1
    mu = (b - 0.5 * s ** 2) / s ** 2
    lam = np.sqrt(mu ** 2 + 2 * r / s ** 2)
    x1 = np.log(S / K) / raiz + (1 + mu) * raiz
    x2 = np.log(S / H) / raiz + (1 + mu) * raiz
    y1 = np.log(H ** 2 / (S * K)) / raiz + (1 + mu) * raiz
    y2 = np.log(H / S) / raiz + (1 + mu) * raiz
    z = np.log(H / S) / raiz + lam * raiz
    cq, dr = np.exp((b - r) * T), np.exp(-r * T)
    A = phi * S * cq * N(phi * x1) - phi * K * dr * N(phi * x1 - phi * raiz)
    Bq = phi * S * cq * N(phi * x2) - phi * K * dr * N(phi * x2 - phi * raiz)
    C = (phi * S * cq * (H / S) ** (2 * (mu + 1)) * N(eta * y1)
         - phi * K * dr * (H / S) ** (2 * mu) * N(eta * y1 - eta * raiz))
    D = (phi * S * cq * (H / S) ** (2 * (mu + 1)) * N(eta * y2)
         - phi * K * dr * (H / S) ** (2 * mu) * N(eta * y2 - eta * raiz))
    E = rebate * dr * (N(eta * x2 - eta * raiz) - (H / S) ** (2 * mu) * N(eta * y2 - eta * raiz))
    F = rebate * ((H / S) ** (mu + lam) * N(eta * z)
                  + (H / S) ** (mu - lam) * N(eta * z - 2 * eta * lam * raiz))

    K_alto = K >= H
    if accion == "entra":
        tabla = {
            ("call", "abajo"): (C + E) if K_alto else (A - Bq + D + E),
            ("call", "arriba"): (A + E) if K_alto else (Bq - C + D + E),
            ("put", "abajo"): (Bq - C + D + E) if K_alto else (A + E),
            ("put", "arriba"): (A - Bq + D + E) if K_alto else (C + E),
        }
    else:
        tabla = {
            ("call", "abajo"): (A - C + F) if K_alto else (Bq - D + F),
            ("call", "arriba"): F if K_alto else (A - Bq + C - D + F),
            ("put", "abajo"): (A - Bq + C - D + F) if K_alto else F,
            ("put", "arriba"): (Bq - D + F) if K_alto else (A - C + F),
        }
    return float(tabla[(tipo, direccion)])


def prob_tocar(m: Mercado, H, T):
    """P_Q(tocar H antes de T)."""
    S, r, q, b, s = _datos(m, T)
    nu = b - 0.5 * s ** 2
    raiz = s * np.sqrt(T)
    x = np.log(H / S)
    if H < S:
        return float(N((x - nu * T) / raiz) + np.exp(2 * nu * x / s ** 2) * N((x + nu * T) / raiz))
    if H > S:
        return float(N((-x + nu * T) / raiz) + np.exp(2 * nu * x / s ** 2) * N((-x - nu * T) / raiz))
    return 1.0


def un_toque(m: Mercado, H, T, monto=1.0, momento="vencimiento"):
    """One-touch: paga `monto` al tocar H, o en T si se pactó al vencimiento."""
    if momento == "vencimiento":
        return float(monto * m.B(T) * prob_tocar(m, H, T))
    S, r, q, b, s = _datos(m, T)
    if S == H:
        return float(monto)
    raiz = s * np.sqrt(T)
    mu = (b - 0.5 * s ** 2) / s ** 2
    lam = np.sqrt(mu ** 2 + 2 * r / s ** 2)
    eta = 1 if S > H else -1
    z = np.log(H / S) / raiz + lam * raiz
    return float(monto * ((H / S) ** (mu + lam) * N(eta * z)
                          + (H / S) ** (mu - lam) * N(eta * z - 2 * eta * lam * raiz)))


def sin_toque(m, H, T, monto=1.0):
    return float(monto * m.B(T) * (1 - prob_tocar(m, H, T)))


# ---------------------------------------------------------------------------
# Doble barrera: densidad por el método de imágenes
# ---------------------------------------------------------------------------
def _densidad_banda(y, x, l, u, nu, s, T, terminos=12):
    """Densidad de ln S_T con absorción en ln L y ln U."""
    w = u - l
    v = s ** 2 * T
    enes = np.arange(-terminos, terminos + 1)[:, None]
    g = lambda z: np.exp(-z ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    suma = (g(y - x - 2 * enes * w) - g(y + x - 2 * l - 2 * enes * w)).sum(axis=0)
    return np.exp(nu * (y - x) / s ** 2 - nu ** 2 * T / (2 * s ** 2)) * suma


def banda_salida(m: Mercado, L, U, T, payoff):
    """E_Q[B · payoff(S_T) · 1{L < S_t < U para todo t}] con cuadratura de Gauss-Legendre."""
    S, r, q, b, s = _datos(m, T)
    if not (L < S < U):
        return 0.0
    x, l, u = np.log(S), np.log(L), np.log(U)
    nodos, pesos = np.polynomial.legendre.leggauss(400)
    y = 0.5 * (u - l) * nodos + 0.5 * (u + l)
    dens = _densidad_banda(y, x, l, u, b - 0.5 * s ** 2, s, T)
    return float(m.B(T) * 0.5 * (u - l) * np.sum(pesos * dens * payoff(np.exp(y))))


def doble_barrera(m: Mercado, K, L, U, T, tipo="call", accion="sale"):
    pf = (lambda S: np.maximum(S - K, 0)) if tipo == "call" else (lambda S: np.maximum(K - S, 0))
    fuera = banda_salida(m, L, U, T, pf)
    return fuera if accion == "sale" else float(europea(m, K, T, tipo)) - fuera


def opcion_banda(m: Mercado, L, U, T, monto=None):
    """La de las notas: X = S(0)·1{B1 < S(t) < B2 para todo t}."""
    monto = m.S0 if monto is None else monto
    return banda_salida(m, L, U, T, lambda S: monto * np.ones_like(S))


def acumulador_rango(m: Mercado, L, U, T, cupon, obs=252):
    """Range accrual: cupon·(fracción de días en [L, U]), pagado en T."""
    tiempos = (np.arange(obs) + 1) * T / obs
    dentro = 0.0
    S, r, q, b, s = _datos(m, T)
    for t in tiempos:
        raiz = s * np.sqrt(t)
        d = lambda K: (np.log(S / K) + (b - 0.5 * s ** 2) * t) / raiz
        dentro += N(d(L)) - N(d(U))
    return float(m.B(T) * cupon * dentro / obs)


# ---------------------------------------------------------------------------
# Asiáticas geométricas
# ---------------------------------------------------------------------------
def asiatica_geo_discreta(m: Mercado, K, T, fechas, tipo="call"):
    """Promedio geométrico de S(iT/n), i = 1..n."""
    S, r, q, b, s = _datos(m, T)
    media = np.log(S) + (b - 0.5 * s ** 2) * T * (fechas + 1) / (2 * fechas)
    var = s ** 2 * T * (fechas + 1) * (2 * fechas + 1) / (6 * fechas ** 2)
    return float(_black_lognormal(np.exp(media + 0.5 * var), K, np.sqrt(var), m.B(T), tipo))


def asiatica_geo_continua(m: Mercado, K, T, tipo="call"):
    S, r, q, b, s = _datos(m, T)
    media = np.log(S) + 0.5 * (b - 0.5 * s ** 2) * T
    var = s ** 2 * T / 3
    return float(_black_lognormal(np.exp(media + 0.5 * var), K, np.sqrt(var), m.B(T), tipo))


# ---------------------------------------------------------------------------
# Lookback (monitoreo continuo)
# ---------------------------------------------------------------------------
def _b_seguro(b):
    return b if abs(b) > 1e-7 else 1e-7


def lookback_flotante(m: Mercado, T, tipo="call", extremo=None):
    """Call: S_T - mín; put: máx - S_T (Goldman-Sosin-Gatto)."""
    S, r, q, b0, s = _datos(m, T)
    b = _b_seguro(b0)
    E = S if extremo is None else extremo
    raiz = s * np.sqrt(T)
    dr, cq = np.exp(-r * T), np.exp((b - r) * T)
    a1 = (np.log(S / E) + (b + 0.5 * s ** 2) * T) / raiz
    a2 = a1 - raiz
    k = S * dr * s ** 2 / (2 * b)
    pot = (S / E) ** (-2 * b / s ** 2)
    if tipo == "call":
        return float(S * cq * N(a1) - E * dr * N(a2)
                     + k * (pot * N(-a1 + 2 * b * np.sqrt(T) / s) - np.exp(b * T) * N(-a1)))
    return float(E * dr * N(-a2) - S * cq * N(-a1)
                 + k * (-pot * N(a1 - 2 * b * np.sqrt(T) / s) + np.exp(b * T) * N(a1)))


def lookback_fijo(m: Mercado, K, T, tipo="call", extremo=None):
    """Call: máx - K; put: K - mín (Conze-Viswanathan)."""
    S, r, q, b0, s = _datos(m, T)
    b = _b_seguro(b0)
    E = S if extremo is None else extremo
    raiz = s * np.sqrt(T)
    rt = np.sqrt(T)
    dr, cq = np.exp(-r * T), np.exp((b - r) * T)
    k = S * dr * s ** 2 / (2 * b)
    d = lambda X: (np.log(S / X) + (b + 0.5 * s ** 2) * T) / raiz
    pot = lambda X: (S / X) ** (-2 * b / s ** 2)
    if tipo == "call":
        if K >= E:
            d1 = d(K)
            return float(S * cq * N(d1) - K * dr * N(d1 - raiz)
                         + k * (-pot(K) * N(d1 - 2 * b * rt / s) + np.exp(b * T) * N(d1)))
        e1 = d(E)
        return float(dr * (E - K) + S * cq * N(e1) - E * dr * N(e1 - raiz)
                     + k * (-pot(E) * N(e1 - 2 * b * rt / s) + np.exp(b * T) * N(e1)))
    if K <= E:
        d1 = d(K)
        return float(K * dr * N(-d1 + raiz) - S * cq * N(-d1)
                     + k * (pot(K) * N(-d1 + 2 * b * rt / s) - np.exp(b * T) * N(-d1)))
    f1 = d(E)
    return float(dr * (K - E) - S * cq * N(-f1) + E * dr * N(-f1 + raiz)
                 + k * (pot(E) * N(-f1 + 2 * b * rt / s) - np.exp(b * T) * N(-f1)))


def lookback_maximo(m: Mercado, T):
    """X = máx{S(t)}: el lookback de las notas, en monitoreo continuo."""
    return float(lookback_flotante(m, T, "put", m.S0) + m.prepago(T))


# ---------------------------------------------------------------------------
# Opciones sobre el tiempo
# ---------------------------------------------------------------------------
def selectora(m: Mercado, K, t_elige, T):
    """Chooser simple: en t_elige se decide si es call o put, mismo K y T."""
    S, r, q, b, s = _datos(m, T)
    d = (np.log(S / K) + (b + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    y = (np.log(S / K) + b * T + 0.5 * s ** 2 * t_elige) / (s * np.sqrt(t_elige))
    cq, dr = np.exp((b - r) * T), np.exp(-r * T)
    return float(S * cq * N(d) - K * dr * N(d - s * np.sqrt(T))
                 - S * cq * N(-y) + K * dr * N(-y + s * np.sqrt(t_elige)))


def compuesta(m: Mercado, K_ext, t1, K_int, T2, ext="call", inter="call"):
    """Opción `ext` con vencimiento t1 y strike K_ext sobre una opción `inter`.

    Es la integral de Geske evaluada numéricamente sobre S(t1).
    """
    S, r, q, b, s = _datos(m, T2)
    tau = T2 - t1

    def valor_interno(St):
        return black(St * np.exp(-q * tau), K_int * np.exp(-r * tau), s, tau, inter)

    def integrando(z):
        St = S * np.exp((b - 0.5 * s ** 2) * t1 + s * np.sqrt(t1) * z)
        v = valor_interno(St)
        return max(v - K_ext, 0.0) if ext == "call" else max(K_ext - v, 0.0)

    nodos = np.linspace(-8, 8, 8001)
    vals = np.array([integrando(z) for z in nodos]) * n(nodos)
    return float(np.exp(-r * t1) * np.trapezoid(vals, nodos))


def inicio_diferido(m: Mercado, alfa, t_ini, T, tipo="call"):
    """Forward-start: en t_ini se fija K = alfa·S(t_ini)."""
    tau = T - t_ini
    S, r, q, b, s = _datos(m, T)
    unidad = black(np.exp(-q * tau), alfa * np.exp(-r * tau), s, tau, tipo)
    return float(S * np.exp(-q * t_ini) * unidad)


def trinquete(m: Mercado, T, periodos, alfa=1.0):
    """Cliquet sin topes, por unidad de nominal: B(0,T)·Σ E[(S_{i+1}/S_i - alfa)+], pagado en T."""
    tau = T / periodos
    r, q, s = m.rc, m.qc, m.sigma
    unidad = black(np.exp(-q * tau), alfa * np.exp(-r * tau), s, tau, "call") * np.exp(r * tau)
    return float(m.B(T) * periodos * unidad)


# ---------------------------------------------------------------------------
# Dos activos
# ---------------------------------------------------------------------------
def margrabe(S1, S2, q1, q2, s1, s2, rho, T):
    """Intercambio: (S1 - S2)+."""
    s = np.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    return float(black(S1 * np.exp(-q1 * T), S2 * np.exp(-q2 * T), s, T, "call"))


def kirk(S1, S2, K, r, q1, q2, s1, s2, rho, T):
    """Aproximación de Kirk para el spread (S1 - S2 - K)+."""
    F1, F2 = S1 * np.exp((r - q1) * T), S2 * np.exp((r - q2) * T)
    w = F2 / (F2 + K)
    s = np.sqrt(s1 ** 2 - 2 * rho * s1 * s2 * w + (s2 * w) ** 2)
    return float(np.exp(-r * T) * _black_lognormal(F1, F2 + K, s * np.sqrt(T), 1.0, "call"))
