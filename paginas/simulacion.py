"""Simulación y constructor de contratos a la medida.

La misma función `evaluar` sirve para trayectorias de Monte Carlo y para las
2^N trayectorias exactas de un árbol binomial.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import numpy as np

from .base import Mercado
from .exoticos import asiatica_geo_discreta
from .vanilla import enumerar_trayectorias

BETA_BGK = 0.5826   # corrección de Broadie-Glasserman-Kou


# ---------------------------------------------------------------------------
# Trayectorias
# ---------------------------------------------------------------------------
def simular(m: Mercado, T, pasos, n, semilla=7):
    """Trayectorias con antitéticas. S = S* + valor presente de los pagos que faltan."""
    rng = np.random.default_rng(semilla)
    mitad = (n + 1) // 2
    delta = T / pasos
    tiempos = np.arange(pasos + 1) * delta
    # Todo se hace sobre la misma matriz para no duplicar memoria.
    S = np.zeros((n, pasos + 1))
    S[:mitad, 1:] = rng.standard_normal((mitad, pasos))
    S[mitad:, 1:] = -S[:n - mitad, 1:]
    S[:, 1:] *= m.sigma * np.sqrt(delta)
    S[:, 1:] += (m.b - 0.5 * m.sigma ** 2) * delta
    np.cumsum(S, axis=1, out=S)
    np.exp(S, out=S)
    S *= m.S_estrella(T)
    for ti, di in m.pagos_hasta(T):
        S += np.where(tiempos < ti - 1e-12, di * np.exp(-m.b * (ti - tiempos)), 0.0)
    return tiempos, S


# ---------------------------------------------------------------------------
# Expresiones de payoff seguras
# ---------------------------------------------------------------------------
_FUNCIONES = {"max": np.maximum, "min": np.minimum, "abs": np.abs, "exp": np.exp,
              "log": np.log, "ln": np.log, "sqrt": np.sqrt,
              "si": lambda c, a, b: np.where(c, a, b)}


def _eval_nodo(nodo, env):
    if isinstance(nodo, ast.Expression):
        return _eval_nodo(nodo.body, env)
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
        return float(nodo.value)
    if isinstance(nodo, ast.Name):
        if nodo.id in env:
            return env[nodo.id]
        raise ValueError(f"No conozco la variable «{nodo.id}».")
    if isinstance(nodo, ast.BinOp):
        a, b = _eval_nodo(nodo.left, env), _eval_nodo(nodo.right, env)
        ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
               ast.Div: lambda: a / b, ast.Pow: lambda: a ** b,
               ast.BitAnd: lambda: np.logical_and(a, b), ast.BitOr: lambda: np.logical_or(a, b)}
        if type(nodo.op) in ops:
            return ops[type(nodo.op)]()
    if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, (ast.USub, ast.UAdd)):
        v = _eval_nodo(nodo.operand, env)
        return -v if isinstance(nodo.op, ast.USub) else v
    if isinstance(nodo, ast.Compare) and len(nodo.ops) == 1:
        a, b = _eval_nodo(nodo.left, env), _eval_nodo(nodo.comparators[0], env)
        ops = {ast.Gt: a > b, ast.GtE: a >= b, ast.Lt: a < b, ast.LtE: a <= b}
        if type(nodo.ops[0]) in ops:
            return ops[type(nodo.ops[0])]
    if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) and nodo.func.id in _FUNCIONES:
        args = [_eval_nodo(a, env) for a in nodo.args]
        f = _FUNCIONES[nodo.func.id]
        if nodo.func.id in ("max", "min"):
            r = args[0]
            for a in args[1:]:
                r = f(r, a)
            return r
        return f(*args)
    raise ValueError("La expresión tiene algo que no puedo evaluar. "
                     "Usa + - * / ^, comparaciones y max, min, abs, exp, log, sqrt, si.")


def evaluar_expresion(texto: str, env: dict):
    try:
        arbolito = ast.parse(texto.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ValueError("La expresión está incompleta o mal escrita.")
    return _eval_nodo(arbolito, env)


# ---------------------------------------------------------------------------
# Contrato
# ---------------------------------------------------------------------------
@dataclass
class Contrato:
    base: str = "call"          # call put straddle digital_call digital_put digital_activo
                                # forward monto call_spread put_spread expresion
    referencia: str = "final"   # final prom_ar prom_geo max min
    strike_tipo: str = "fijo"   # fijo prom_ar prom_geo max min adelantado
    K: float = 100.0
    K2: float = 110.0
    monto: float = 1.0
    potencia: float = 1.0
    t_prom: float = 0.0         # inicio de la ventana de promedio
    prom_con_S0: bool = False   # incluye S(0) en el promedio, como en la sesión 8
    t_adelantado: float = 0.0
    alfa_adelantado: float = 1.0
    participacion: float = 1.0
    tope: float | None = None
    piso: float | None = None
    nominal: float = 1.0
    barrera: str = "ninguna"    # ninguna sale entra
    L: float | None = None
    U: float | None = None
    t_bar_ini: float = 0.0
    t_bar_fin: float | None = None
    bgk: bool = False
    rebate: float = 0.0
    rebate_momento: str = "toque"
    autollamado: float | None = None
    autollamado_cupon: float = 0.0
    autollamado_fechas: tuple = ()
    rango_L: float | None = None
    rango_U: float | None = None
    rango_cupon: float = 0.0
    ejercicio: str = "europeo"  # europeo americano bermuda
    fechas_ej: tuple = ()
    expresion: str = ""


@dataclass
class Resultado:
    precio: float
    ee: float
    ic: tuple
    p_toque: float | None = None
    p_autollamado: float | None = None
    p_temprano: float | None = None
    t_medio: float | None = None
    vp: np.ndarray | None = field(default=None, repr=False)


def _estadisticas(c: Contrato, tiempos, S, k):
    """Promedios y extremos hasta el índice k, por trayectoria."""
    ini = max(int(np.searchsorted(tiempos, c.t_prom - 1e-12)), 1)
    ini = 0 if c.prom_con_S0 and c.t_prom <= 0 else min(ini, max(k, 1))
    v = S[:, ini:k + 1] if k >= 1 else S[:, :1]
    return {"final": S[:, k], "prom_ar": v.mean(axis=1), "prom_geo": np.exp(np.log(v).mean(axis=1)),
            "max": S[:, :k + 1].max(axis=1), "min": S[:, :k + 1].min(axis=1)}


def _payoff(c: Contrato, tiempos, S, k, S0, T):
    e = _estadisticas(c, tiempos, S, k)
    X = e[c.referencia]
    if c.potencia != 1.0:
        X = X ** c.potencia
    if c.strike_tipo == "fijo":
        K = c.K
    elif c.strike_tipo == "adelantado":
        K = c.alfa_adelantado * S[:, int(np.searchsorted(tiempos, c.t_adelantado - 1e-12))]
    else:
        K = e[c.strike_tipo]
    b = c.base
    if b == "call":
        v = np.maximum(X - K, 0)
    elif b == "put":
        v = np.maximum(K - X, 0)
    elif b == "straddle":
        v = np.abs(X - K)
    elif b == "digital_call":
        v = c.monto * (X > K)
    elif b == "digital_put":
        v = c.monto * (X < K)
    elif b == "digital_activo":
        v = S[:, k] * (X > K)
    elif b == "forward":
        v = X - K
    elif b == "monto":
        v = c.monto * np.ones(S.shape[0])
    elif b == "call_spread":
        v = np.clip(X - K, 0, c.K2 - c.K)
    elif b == "put_spread":
        v = np.clip(c.K2 - X, 0, c.K2 - c.K)
    elif b == "expresion":
        env = {"S": S[:, k], "A": e["prom_ar"], "G": e["prom_geo"], "M": e["max"], "m": e["min"],
               "K": c.K, "K2": c.K2, "S0": S0, "T": T}
        v = np.broadcast_to(np.asarray(evaluar_expresion(c.expresion, env), dtype=float), S[:, k].shape)
    else:
        raise ValueError(b)
    v = c.participacion * v
    if c.tope is not None:
        v = np.minimum(v, c.tope)
    if c.piso is not None:
        v = np.maximum(v, c.piso)
    return c.nominal * v


def _primer_toque(c: Contrato, m: Mercado, tiempos, S):
    """Índice del primer cruce de barrera; -1 si no hubo."""
    n = S.shape[0]
    if c.barrera == "ninguna" or (c.L is None and c.U is None):
        return np.full(n, -1)
    delta = tiempos[1] - tiempos[0]
    ajuste = np.exp(BETA_BGK * m.sigma * np.sqrt(delta)) if c.bgk else 1.0
    ventana = (tiempos >= c.t_bar_ini - 1e-12)
    if c.t_bar_fin is not None:
        ventana &= tiempos <= c.t_bar_fin + 1e-12
    toca = np.zeros_like(S, dtype=bool)
    if c.L is not None:
        toca |= S <= c.L * ajuste
    if c.U is not None:
        toca |= S >= c.U / ajuste
    toca[:, ~ventana] = False
    hubo = toca.any(axis=1)
    return np.where(hubo, toca.argmax(axis=1), -1)


def evaluar(c: Contrato, m: Mercado, tiempos, S, pesos=None) -> Resultado:
    n, npt = S.shape
    ult = npt - 1
    T = tiempos[-1]
    S0 = float(S[0, 0])
    dsc = np.exp(-m.rc * tiempos)
    w = np.ones(n) / n if pesos is None else pesos

    terminal = _payoff(c, tiempos, S, ult, S0, T)
    toque = _primer_toque(c, m, tiempos, S)
    hubo = toque >= 0
    idx = np.full(n, ult)
    monto = terminal.copy()

    if c.barrera == "sale":
        monto[hubo] = c.rebate
        if c.rebate_momento == "toque":
            idx[hubo] = toque[hubo]
    elif c.barrera == "entra":
        monto[~hubo] = c.rebate

    extra = np.zeros(n)
    if c.rango_cupon and (c.rango_L is not None or c.rango_U is not None):
        lo = -np.inf if c.rango_L is None else c.rango_L
        hi = np.inf if c.rango_U is None else c.rango_U
        dentro = (S[:, 1:] >= lo) & (S[:, 1:] <= hi)
        if c.barrera == "sale":
            vivo = np.where(hubo, toque, ult + 1)[:, None] > np.arange(1, ult + 1)
            dentro &= vivo
        extra = c.rango_cupon * dentro.mean(axis=1) * dsc[ult]

    p_auto = None
    if c.autollamado is not None and c.autollamado_fechas:
        kk = sorted({int(np.argmin(np.abs(tiempos - t))) for t in c.autollamado_fechas})
        llamado = np.zeros(n, bool)
        vivo = ~hubo if c.barrera == "sale" else np.ones(n, bool)
        for orden, k in enumerate(kk, start=1):
            activar = vivo & ~llamado & (S[:, k] >= c.autollamado)
            if c.barrera == "sale":
                activar &= (~hubo) | (toque > k)
            monto[activar] = c.nominal * c.monto + orden * c.autollamado_cupon
            idx[activar] = k
            llamado |= activar
        p_auto = float(np.sum(w[llamado]))

    p_temp = None
    if c.ejercicio != "europeo":
        monto, idx, p_temp = _lsm(c, m, tiempos, S, monto, idx, toque, S0, T)

    vp = monto * dsc[idx] + extra
    precio = float(np.sum(w * vp))
    if pesos is None:
        ee = float(vp.std(ddof=1) / np.sqrt(n))
    else:
        ee = 0.0
    if c.ejercicio == "americano":
        p0 = float(_payoff(c, tiempos, S[:1], 0, S0, T)[0]) if c.strike_tipo == "fijo" and c.referencia == "final" else 0.0
        precio = max(precio, p0)
    return Resultado(precio, ee, (precio - 1.96 * ee, precio + 1.96 * ee),
                     float(np.sum(w[hubo])) if c.barrera != "ninguna" else None,
                     p_auto, p_temp, float(np.sum(w * tiempos[idx])), vp)


def _lsm(c: Contrato, m, tiempos, S, monto, idx, toque, S0, T):
    """Longstaff-Schwartz. El precio resultante es una cota inferior."""
    ult = S.shape[1] - 1
    if c.ejercicio == "americano":
        fechas = list(range(1, ult))
    else:
        fechas = sorted({int(np.argmin(np.abs(tiempos - t))) for t in c.fechas_ej})
        fechas = [k for k in fechas if 0 < k < ult]
    monto, idx = monto.copy(), idx.copy()
    temprano = np.zeros(S.shape[0], bool)
    hubo = toque >= 0
    for k in reversed(fechas):
        X = _payoff(c, tiempos, S, k, S0, T)
        if c.barrera == "sale":
            X = np.where(hubo & (toque <= k), 0.0, X)
        elif c.barrera == "entra":
            X = np.where(hubo & (toque <= k), X, 0.0)
        itm = X > 0
        if itm.sum() < 20:
            continue
        cont = monto[itm] * np.exp(-m.rc * (tiempos[idx[itm]] - tiempos[k]))
        s = S[itm, k] / S0
        ref = _estadisticas(c, tiempos, S[itm], k)[c.referencia] / S0
        columnas = [np.ones_like(s), s, s ** 2, s ** 3]
        if c.referencia != "final" or c.strike_tipo != "fijo":
            columnas += [ref, ref ** 2, s * ref]
        A = np.column_stack(columnas)
        coef = np.linalg.lstsq(A, cont, rcond=None)[0]
        ejercer = X[itm] > A @ coef
        pos = np.where(itm)[0][ejercer]
        monto[pos] = X[pos]
        idx[pos] = k
        temprano[pos] = True
    return monto, idx, float(temprano.mean())


def valuar_mc(c: Contrato, m: Mercado, T, pasos, n, semilla=7) -> Resultado:
    tiempos, S = simular(m, T, pasos, n, semilla)
    return evaluar(c, m, tiempos, S)


def valuar_arbol(c: Contrato, m: Mercado, T, pasos, metodo="crr") -> Resultado:
    """Suma exacta sobre las 2^N trayectorias del árbol."""
    tiempos, S, pesos = enumerar_trayectorias(m, T, pasos, metodo)
    return evaluar(c, m, tiempos, S, pesos)


def griegas_mc(c: Contrato, m: Mercado, T, pasos, n, semilla=7) -> dict:
    """Con números aleatorios comunes, para reducir el ruido de las diferencias."""
    f = lambda mm: valuar_mc(c, mm, T, pasos, n, semilla).precio
    h = 0.01 * m.S0
    v, vu, vd = f(m), f(m.con(S0=m.S0 + h)), f(m.con(S0=m.S0 - h))
    vega = (f(m.con(sigma=m.sigma + 0.01)) - f(m.con(sigma=m.sigma - 0.01))) / 0.02
    rho = (f(m.con(r=m.r + 0.001)) - f(m.con(r=m.r - 0.001))) / 0.002
    return {"Delta": (vu - vd) / (2 * h), "Gamma": (vu - 2 * v + vd) / h ** 2, "Vega": vega, "Rho": rho}


# ---------------------------------------------------------------------------
# Exóticos con simulación propia
# ---------------------------------------------------------------------------
def asiatica_mc(m: Mercado, K, T, fechas, tipo="call", clase="precio", n=100_000, semilla=7) -> Resultado:
    """Promedio aritmético. La geométrica sirve de variable de control si no hay pagos discretos."""
    tiempos, S = simular(m, T, fechas, n, semilla)
    A = S[:, 1:].mean(axis=1)
    fin = S[:, -1]
    B = m.B(T)
    signo = 1 if tipo == "call" else -1
    if clase == "precio":
        Y = B * np.maximum(signo * (A - K), 0)
    else:
        Y = B * np.maximum(signo * (fin - A), 0)
    if clase == "precio" and not m.pagos:
        G = np.exp(np.log(S[:, 1:]).mean(axis=1))
        Z = B * np.maximum(signo * (G - K), 0)
        Ez = asiatica_geo_discreta(m, K, T, fechas, tipo)
        var = np.var(Z, ddof=1)
        beta = np.cov(Y, Z, ddof=1)[0, 1] / var if var > 0 else 0.0
        Y = Y - beta * (Z - Ez)
    p, ee = float(Y.mean()), float(Y.std(ddof=1) / np.sqrt(n))
    return Resultado(p, ee, (p - 1.96 * ee, p + 1.96 * ee))


def trinquete_mc(m: Mercado, T, periodos, tope_local=None, piso_local=0.0, piso_global=None,
                 tope_global=None, nominal=1.0, n=100_000, semilla=7) -> Resultado:
    """Cliquet: suma de rendimientos periódicos acotados."""
    tiempos, S = simular(m, T, periodos, n, semilla)
    R = S[:, 1:] / S[:, :-1] - 1
    if tope_local is not None:
        R = np.minimum(R, tope_local)
    R = np.maximum(R, piso_local)
    suma = R.sum(axis=1)
    if piso_global is not None:
        suma = np.maximum(suma, piso_global)
    if tope_global is not None:
        suma = np.minimum(suma, tope_global)
    Y = m.B(T) * nominal * suma
    p, ee = float(Y.mean()), float(Y.std(ddof=1) / np.sqrt(n))
    return Resultado(p, ee, (p - 1.96 * ee, p + 1.96 * ee))


def multiactivo_mc(S0s, sigmas, qs, r, rho, T, pago, n=200_000, semilla=7) -> Resultado:
    """`pago(ST)` recibe una matriz (trayectorias, activos). Correlación constante rho."""
    S0s, sigmas, qs = map(lambda v: np.asarray(v, float), (S0s, sigmas, qs))
    d = len(S0s)
    R = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(semilla)
    Z = rng.standard_normal(((n + 1) // 2, d))
    Z = np.vstack([Z, -Z])[:n] @ L.T
    ST = S0s * np.exp((r - qs - 0.5 * sigmas ** 2) * T + sigmas * np.sqrt(T) * Z)
    Y = np.exp(-r * T) * pago(ST)
    p, ee = float(Y.mean()), float(Y.std(ddof=1) / np.sqrt(n))
    return Resultado(p, ee, (p - 1.96 * ee, p + 1.96 * ee))
