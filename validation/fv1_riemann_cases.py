"""Physical inputs and exact solutions for the FV1 flux investigation."""

from dataclasses import dataclass

import numpy as np

from validation.analytical_convergence import star_residual
from scipy.optimize import brentq


@dataclass
class Domain:
    dx: float
    bed: np.ndarray
    initial: np.ndarray
    duration: float
    rain: float | np.ndarray = 0.
    manning: float = 0.
    stage_x: tuple | None = None
    stage_y: float | None = None


def coordinates(*, length, width, dx):
    x = (np.arange(round(length / dx)) + .5) * dx - length / 2.
    y = (np.arange(round(width / dx)) + .5) * dx - width / 2.
    return np.meshgrid(x, y)


def stoker_star(*, left, right, gravity):
    def residual(depth):
        return star_residual(depth=depth, left_depth=left, right_depth=right, gravity=gravity)

    h = brentq(f=residual, a=right, b=left, xtol=1.e-14)
    u = 2. * (np.sqrt(gravity * left) - np.sqrt(gravity * h))
    return h, u, h * u / (h - right)


def exact_strip(*, name, x, time, gravity):
    """Point solution, breakpoints for integration, and feature coordinates."""
    v = np.zeros_like(x)
    features = {}
    if name in ('ritter', 'stoker'):
        left, right = 2., (.4 if name == 'stoker' else 0.)
        cl = np.sqrt(gravity * left)
        hs, us, shock = stoker_star(left=left, right=right, gravity=gravity) if right else (0., 2. * cl, 2. * cl)
        head, tail = -cl * time, (us - np.sqrt(gravity * hs)) * time
        edge = shock * time
        h = np.where(x < head, left, np.where(x < edge, hs, right))
        u = np.where((x >= head) & (x < edge), us, 0.)
        if time:
            speed = x / time
            fan = (x >= head) & (x < tail)
            h = np.where(fan, (2. * cl - speed)**2 / (9. * gravity), h)
            u = np.where(fan, 2. * (cl + speed) / 3., u)
        features = dict(rarefaction_head=head, rarefaction_tail=tail)
        features['shock' if right else 'front'] = edge
        breaks = (head, tail, edge)
    elif name == 'thacker':
        alpha, amplitude, halfwidth = .0001, 48., 120.
        omega = np.sqrt(2. * gravity * alpha)
        centre = amplitude * np.cos(omega * time)
        u = np.full_like(x, -amplitude * omega * np.sin(omega * time))
        h = np.maximum(0., alpha * (halfwidth**2 - (x - centre)**2))
        breaks = (centre - halfwidth, centre + halfwidth)
        features = dict(left_front=breaks[0], right_front=breaks[1])
    elif name == 'manufactured':
        # Steady u and exponential h. R=(hu)_x; z=-h-u²/g log(h/H).
        h = .8 * np.exp(x / 640.)
        u = np.full_like(x, 1.25)
        breaks = ()
    elif name in ('shear', 'stationary_shear'):
        speed = 1.5 if name == 'shear' else 0.
        h = np.ones_like(x)
        u = np.full_like(x, speed)
        breaks = (time * speed,)
        v = np.where(x < breaks[0], 1.8, -.7)
        features = dict(shear=breaks[0])
    elif name in ('jump', 'jump_offset'):
        h1, froude = .6, 2.25
        h2 = h1 * (np.sqrt(1. + 8. * froude**2) - 1.) / 2.
        discharge = froude * h1 * np.sqrt(gravity * h1)
        breaks = (0. if name == 'jump' else 13.75,)
        h = np.where(x < breaks[0], h1, h2)
        u = discharge / h
        features = dict(shock=breaks[0])
    else:
        raise ValueError(f'Unknown analytical strip {name}')
    return np.stack((h, h * u, h * v), axis=-1), breaks, features


def cell_averages(*, name, x, dx, time, gravity):
    _, breaks, _ = exact_strip(name=name, x=x, time=time, gravity=gravity)
    abscissae, weights = np.polynomial.legendre.leggauss(deg=8)
    west, east = x - dx / 2., x + dx / 2.
    edges = [west, *[np.clip(point, west, east) for point in sorted(set(breaks))], east]
    result = np.zeros((*x.shape, 3))
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        width = (right - left) / 2.
        points = (right + left)[..., None] / 2. + width[..., None] * abscissae
        state, _, _ = exact_strip(name=name, x=points, time=time, gravity=gravity)
        result += (state * weights[:, None]).sum(axis=-2) * (width / dx)[..., None]
    return result


def analytical_domain(*, name, dx, length, width, gravity):
    if name == 'lake':
        x, y = coordinates(length=length, width=length / 2., dx=dx)
        bed = (.32 * np.sin(x / 51.) + .42 * np.cos(y / 28.) + .38 * np.sin((x + y) / 23.)
               + .11 * np.cos((2. * x - y) / 11.))
        state = np.stack((np.maximum(0., .1 - bed), np.zeros_like(x), np.zeros_like(x)), axis=-1)
        return Domain(dx=dx, bed=bed, initial=state, duration=32.)
    x, _ = coordinates(length=length, width=width, dx=dx)
    state = cell_averages(name=name, x=x, dx=dx, time=0., gravity=gravity)
    bed, rain = np.zeros_like(x), 0.
    stage_x, stage_y = None, None
    if name == 'thacker':
        bed = .0001 * x**2
    if name == 'manufactured':
        h = exact_strip(name=name, x=x, time=0., gravity=gravity)[0][..., 0]
        bed = -h - 1.25**2 / gravity * np.log(h / .8)
        rain = 1.25 * state[..., 0] / 640.
        stage_x = tuple((bed + state[..., 0])[0, [0, -1]])
    if name in ('shear', 'stationary_shear'):
        stage_x = (1., 1.)
        stage_y = 1.
    if name in ('jump', 'jump_offset'):
        stage_x = tuple(state[0, [0, -1], 0])
    durations = dict(ritter=32., stoker=32., thacker=32., manufactured=32.,
                     shear=32., stationary_shear=32., jump=16., jump_offset=16.)
    return Domain(dx=dx, bed=bed, initial=state, duration=durations[name], rain=rain, stage_x=stage_x, stage_y=stage_y)


def synthetic_domain(*, name, dx, length, width):
    x, y = coordinates(length=length, width=width, dx=dx)
    s, t = x / length, y / width
    bed = -.55 * s
    h = .44 + .12 * np.exp(-((s + .16)**2 + t**2) / .07)
    u, v, rain, n = .35, 0., 0., .025
    if name in ('irregular', 'rainfall'):
        bed = bed + .27 * np.cos(11. * s + 8. * t) + .35 * np.sin(9. * t - 5. * s) + .16 * np.cos(20. * s - 2. * t)
        h = np.maximum(0., .11 - bed) + .22 * np.exp(-((s + .2)**2 + (t + .13)**2) / .032)
        u = 0.
        if name == 'rainfall':
            h = np.zeros_like(x)
            rain = np.maximum(0., .003 * (np.cos(2. * np.pi * t) + .4 * np.cos(2. * np.pi * s) - .2))
    elif name == 'steep':
        bed = -3.4 * s + 1.25 * t**2 + .26 * np.exp(-((s - .12)**2 + t**2) / .014)
        h = .38 * np.exp(-9. * t**2) * (1. + .28 * np.sin(6. * s))
        u = 3.4 + .2 * np.cos(8. * s)
    elif name == 'expansion':
        channel = .14 + .17 * (1. + np.tanh(15. * s)) / 2.
        bed = bed + 1.4 * (1. + np.tanh(22. * (np.abs(t) - channel))) / 2.
        stage = .22 + .64 * (1. - np.tanh(22. * (s + .21))) / 2.
        h = np.maximum(0., stage - bed)
        u = 0.
    elif name != 'plane':
        raise ValueError(f'Unknown synthetic domain {name}')
    state = np.stack((h, h * u, h * v), axis=-1)
    return Domain(dx=dx, bed=bed, initial=state, duration=16., rain=rain, manning=n)
