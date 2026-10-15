import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import CheckButtons, Button
from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos
from skyfield import almanac

# ---------- Settings ----------
parser = argparse.ArgumentParser(description='Live night sky simulator')
parser.add_argument('--lat', type=float, default=51.4779,
                    help='observer latitude in degrees north (default: Greenwich)')
parser.add_argument('--lon', type=float, default=-0.0015,
                    help='observer longitude in degrees east, negative = west')
parser.add_argument('--mag-limit', type=float, default=4.0,
                    help='faintest star magnitude to draw (lower = fewer stars)')
args = parser.parse_args()
LAT, LON = args.lat, args.lon        # change to your coordinates
MAG_LIMIT = args.mag_limit                # lower = fewer stars drawn
SHOW_CONSTELLATIONS = True      # starting state of the checkbox
UPDATE_SECONDS = 5              # redraw interval at 1x speed
TICK_MS = 1000                  # timer tick; redraws every tick when sped up
ISS_MAX_DAYS = 3                # hide the ISS when simulated time is this far from now
LADDER = [-3600, -600, -60, -10, -1, 1, 10, 60, 600, 3600]   # simulated seconds per real second

ts = load.timescale()
eph = load('de421.bsp')
observer = wgs84.latlon(LAT, LON)
here = eph['earth'] + observer


def project(alt_deg, az_deg):
    r = (90 - np.asarray(alt_deg)) / 90
    th = np.radians(az_deg)
    return -r * np.sin(th), r * np.cos(th)   # north up, east on the left


# ---------- Static data (loaded once) ----------
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)
df = df[df['magnitude'] <= 6.0].copy()
df = df.dropna(subset=['ra_degrees', 'dec_degrees'])
df = df.fillna({'parallax_mas': 0.0, 'ra_mas_per_year': 0.0, 'dec_mas_per_year': 0.0})
df['parallax_mas'] = df['parallax_mas'].clip(lower=0.001)
star_obj = Star.from_dataframe(df)

mag = df['magnitude'].values
shown_mask = mag <= MAG_LIMIT
sizes = (5 - mag) ** 2 * 3 + 2

NAMED = {
    32349: 'Sirius', 30438: 'Canopus', 69673: 'Arcturus', 91262: 'Vega',
    24608: 'Capella', 24436: 'Rigel', 37279: 'Procyon', 27989: 'Betelgeuse',
    97649: 'Altair', 21421: 'Aldebaran', 80763: 'Antares', 65474: 'Spica',
    37826: 'Pollux', 36850: 'Castor', 113368: 'Fomalhaut', 102098: 'Deneb',
    49669: 'Regulus', 11767: 'Polaris', 25336: 'Bellatrix', 54061: 'Dubhe',
    67301: 'Alkaid', 14576: 'Algol', 57632: 'Denebola',
}
named = [(df.index.get_loc(h), n) for h, n in NAMED.items()
         if h in df.index and shown_mask[df.index.get_loc(h)]]

KNOWN = {
    'Ori': 'Orion', 'UMa': 'Ursa Major', 'UMi': 'Ursa Minor', 'Cas': 'Cassiopeia',
    'Cyg': 'Cygnus', 'Lyr': 'Lyra', 'Aql': 'Aquila', 'Sco': 'Scorpius',
    'Leo': 'Leo', 'Tau': 'Taurus', 'Gem': 'Gemini', 'CMa': 'Canis Major',
    'Peg': 'Pegasus', 'And': 'Andromeda', 'Per': 'Perseus', 'Cep': 'Cepheus',
    'Dra': 'Draco', 'Boo': 'Boötes', 'Vir': 'Virgo', 'Sgr': 'Sagittarius',
    'Cnc': 'Cancer', 'Lib': 'Libra', 'Cap': 'Capricornus', 'Aqr': 'Aquarius',
    'Psc': 'Pisces', 'Ari': 'Aries', 'CrB': 'Corona Borealis', 'Her': 'Hercules',
    'Aur': 'Auriga', 'CMi': 'Canis Minor',
}


def load_constellations(cache='modern_skyculture.json'):
    urls = [
        'https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern/index.json',
        'https://raw.githubusercontent.com/Stellarium/stellarium-skycultures/master/modern/index.json',
    ]
    if not os.path.exists(cache):
        for u in urls:
            try:
                with urlopen(u, timeout=20) as r:
                    data = r.read()
                json.loads(data)
                with open(cache, 'wb') as f:
                    f.write(data)
                break
            except Exception as e:
                print(f'  failed: {u} ({e})')
    if not os.path.exists(cache):
        return []
    with open(cache, encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for c in data.get('constellations', []):
        abbr = c['id'].split()[-1]
        edges = []
        for line in c.get('lines', []):
            hips = [h for h in line if isinstance(h, int)]
            edges += list(zip(hips[:-1], hips[1:]))
        result.append((abbr, edges))
    return result


constellations = load_constellations()
print(f'Loaded {len(constellations)} constellations')

cons_idx = {}
for abbr, edges in constellations:
    if abbr in KNOWN:
        pairs = [(df.index.get_loc(a), df.index.get_loc(b))
                 for a, b in edges if a in df.index and b in df.index]
        if pairs:
            arr = np.array(pairs)
            cons_idx[abbr] = (arr[:, 0], arr[:, 1])

DSO = {
    'Pleiades': (3.783, 24.12), 'Hyades': (4.450, 15.87), 'Beehive': (8.673, 19.67),
    'Double Cluster': (2.337, 57.13), 'M13': (16.695, 36.46),
    'Orion Nebula': (5.590, -5.39), 'Andromeda Galaxy': (0.712, 41.27),
}
dso_obj = Star(ra_hours=[v[0] for v in DSO.values()],
               dec_degrees=[v[1] for v in DSO.values()])

# ---------- Planets ----------
PLANET_MAX_SUN_ALT = -3     # planets only drawn when the Sun is below this altitude (deg)
PLANETS = {                 # name: (label, colour, marker size)
    'mercury': ('Mercury', '#b0b0b0', 30),
    'venus':   ('Venus',   '#fff3c4', 60),
    'mars':    ('Mars',    '#ff6a4d', 45),
    'jupiter': ('Jupiter', '#f0d9a8', 65),
    'saturn':  ('Saturn',  '#e8d28a', 50),
}


def get_body(name):
    # de421 stores the outer planets as barycenters, so try both names
    for key in (name, name + ' barycenter'):
        try:
            return eph[key]
        except Exception:
            pass
    raise KeyError(name)


planet_objs = {n: get_body(n) for n in PLANETS}


def sky_phase(sun_alt):
    if sun_alt > 0:
        return 'day'
    if sun_alt > -6:
        return 'civil twilight'
    if sun_alt > -12:
        return 'nautical twilight'
    if sun_alt > -18:
        return 'astronomical twilight'
    return 'night'


# ---------- ISS ----------
TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE'
TLE_FILE = 'iss.tle'


def load_iss():
    stale = not os.path.exists(TLE_FILE) or load.days_old(TLE_FILE) > 1.0
    try:
        return load.tle_file(TLE_URL, filename=TLE_FILE, reload=stale)[0]
    except Exception as e:
        if os.path.exists(TLE_FILE):
            print(f'TLE download failed ({e}); using cached file')
            return load.tle_file(TLE_FILE)[0]
        raise


iss = None
try:
    iss = load_iss()
    t0 = ts.now()
    times, events = iss.find_events(observer, t0, ts.tt_jd(t0.tt + 2), altitude_degrees=10.0)
    for ti, ev in zip(times, events):
        print(['rise', 'peak', 'set '][ev],
              ti.utc_datetime().astimezone().strftime('%a %b %d %H:%M:%S'))
except Exception as e:
    print('Could not load ISS data:', e)

# ---------- Time control state ----------
state = {
    'sim': datetime.now(timezone.utc),   # simulated time
    'wall': time.monotonic(),            # last real-clock reading
    'idx': LADDER.index(1),              # position in LADDER
    'paused': False,
    'last_draw': 0.0,
}


def speed():
    return 0 if state['paused'] else LADDER[state['idx']]


def speed_label():
    if state['paused']:
        return 'Paused'
    s = LADDER[state['idx']]
    return '1x' if s == 1 else f'{s:+d}x'


def advance():
    """Move simulated time forward by (real time elapsed) * speed."""
    now = time.monotonic()
    dt = now - state['wall']
    state['wall'] = now
    state['sim'] += timedelta(seconds=dt * speed())


# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(9, 9), facecolor='black')
rax = fig.add_axes([0.02, 0.02, 0.20, 0.06], facecolor='lightgray')
check = CheckButtons(rax, ['Constellations'], [SHOW_CONSTELLATIONS])
con_artists = []


def draw_sky(t, label):
    """Redraw the whole sky for time t."""
    ax.clear()
    con_artists.clear()
    ax.set_facecolor('black')

        # Sun position drives how dark the sky is
    sa, saz, _ = here.at(t).observe(eph['sun']).apparent().altaz()
    sun_alt, sun_az = sa.degrees, saz.degrees
    f = float(np.clip((sun_alt + 18) / 18, 0, 1))       # 0 = full night, 1 = Sun at/above horizon
    sky = (0.02 + 0.10 * f, 0.03 + 0.20 * f, 0.06 + 0.38 * f)
    star_alpha = float(np.clip(-sun_alt / 12, 0.04, 1.0))

    alt, az, _ = here.at(t).observe(star_obj).apparent().altaz()
    up = alt.degrees > 0
    x, y = project(alt.degrees, az.degrees)

    # Constellation lines + labels
    for abbr, (i1, i2) in cons_idx.items():
        ok = up[i1] & up[i2]
        if not ok.any():
            continue
        p1 = np.column_stack([x[i1][ok], y[i1][ok]])
        p2 = np.column_stack([x[i2][ok], y[i2][ok]])
        lc = LineCollection(np.stack([p1, p2], axis=1), colors='#3d5a80',
                            linewidths=0.9, zorder=1)
        ax.add_collection(lc)
        cx, cy = np.concatenate([p1, p2]).mean(axis=0)
        lab = ax.text(cx, cy, KNOWN[abbr], color='#6c8ebf', fontsize=8,
                      ha='center', zorder=3)
        con_artists.extend([lc, lab])
    for a in con_artists:
        a.set_visible(check.get_status()[0])

    # Stars
    m = shown_mask & up
    ax.scatter(x[m], y[m], s=sizes[m], c='white', alpha=star_alpha, zorder=2)
    for i, name in named:
        if up[i]:
            ax.text(x[i] + 0.015, y[i] + 0.015, name, color='white', fontsize=7, zorder=3)

    # Clusters / deep-sky objects
    dalt, daz, _ = here.at(t).observe(dso_obj).apparent().altaz()
    dx, dy = project(dalt.degrees, daz.degrees)
    for name, px, py, a in zip(DSO, dx, dy, dalt.degrees):
        if a > 0:
            ax.scatter([px], [py], s=120, facecolors='none', edgecolors='orange',
                       lw=1, zorder=2)
            ax.text(px + 0.03, py, name, color='orange', fontsize=7, zorder=3)

    # ISS (only trustworthy near the TLE's date)
    if iss is not None and abs(t.tt - iss.epoch.tt) <= ISS_MAX_DAYS:
        ia, iaz, _ = (iss - observer).at(t).altaz()
        t_track = ts.tt_jd(t.tt + np.arange(0, 90, 0.5) / 1440)
        ta, taz, _ = (iss - observer).at(t_track).altaz()
        tup = ta.degrees > 0
        if tup.any():
            tx, ty = project(ta.degrees[tup], taz.degrees[tup])
            ax.plot(tx, ty, color='cyan', lw=1, alpha=0.6, zorder=2)
        if ia.degrees > 0:
            ix, iy = project(ia.degrees, iaz.degrees)
            ax.scatter([ix], [iy], s=80, c='red', zorder=4)
            ax.text(ix + 0.02, iy + 0.02, 'ISS', color='red', zorder=4)

    # Sun, Moon, planets
    if sun_alt > 0:
        sx, sy = project(sun_alt, sun_az)
        ax.scatter([sx], [sy], s=400, c='gold', zorder=5)
        ax.text(sx + 0.03, sy + 0.03, 'Sun', color='gold', fontsize=8, zorder=5)

    malt, maz, _ = here.at(t).observe(eph['moon']).apparent().altaz()
    if malt.degrees > 0:
        mx, my = project(malt.degrees, maz.degrees)
        try:
            pct = float(almanac.fraction_illuminated(eph, 'moon', t)) * 100
            mlabel = f'Moon {pct:.0f}%'
        except Exception:
            mlabel = 'Moon'
        ax.scatter([mx], [my], s=250, c='#e6e6e6', zorder=5)
        ax.text(mx + 0.03, my + 0.03, mlabel, color='#e6e6e6', fontsize=8, zorder=5)

    if sun_alt < PLANET_MAX_SUN_ALT:
        for name, (plabel, color, size) in PLANETS.items():
            palt, paz, _ = here.at(t).observe(planet_objs[name]).apparent().altaz()
            if palt.degrees > 0:
                px, py = project(palt.degrees, paz.degrees)
                ax.scatter([px], [py], s=size, c=color, zorder=5)
                ax.text(px + 0.02, py + 0.02, plabel, color=color, fontsize=7, zorder=5)

    ax.text(1.05, 0.99, f'Sun {sun_alt:+.0f}\u00b0 \u00b7 {sky_phase(sun_alt)}',
            color='gray', fontsize=8, ha='right')

    # Frame, compass, clock, speed
    ax.add_patch(plt.Circle((0, 0), 1, facecolor=sky, edgecolor='gray', zorder=0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0, 1.03, 'N', color='gray', ha='center')
    ax.text(-1.03, 0, 'E', color='gray', ha='right', va='center')
    ax.text(1.05, -1.03, t.utc_datetime().astimezone().strftime('%Y-%m-%d %H:%M:%S'),
            color='gray', fontsize=8, ha='right')
    ax.text(1.05, 1.03, label, color='gray', fontsize=9, ha='right')


def refresh():
    advance()
    draw_sky(ts.from_datetime(state['sim']), speed_label())
    state['last_draw'] = time.monotonic()
    fig.canvas.draw_idle()


# ---------- Controls ----------
def on_toggle(label):
    vis = check.get_status()[0]
    for a in con_artists:
        a.set_visible(vis)
    fig.canvas.draw_idle()


def on_rewind(event):
    advance()
    state['paused'] = False
    state['idx'] = max(state['idx'] - 1, 0)
    refresh()


def on_forward(event):
    advance()
    state['paused'] = False
    state['idx'] = min(state['idx'] + 1, len(LADDER) - 1)
    refresh()


def on_pause(event):
    advance()
    state['paused'] = not state['paused']
    refresh()


def on_reset(event):
    state['sim'] = datetime.now(timezone.utc)
    state['wall'] = time.monotonic()
    state['idx'] = LADDER.index(1)
    state['paused'] = False
    refresh()


def tick():
    advance()
    sp = speed()
    if sp == 0:
        return
    if sp == 1 and time.monotonic() - state['last_draw'] < UPDATE_SECONDS:
        return
    refresh()


check.on_clicked(on_toggle)

buttons = []   # keep references or matplotlib garbage-collects the widgets
for i, (text, handler) in enumerate([('<<', on_rewind), ('Pause', on_pause),
                                     ('>>', on_forward), ('Reset', on_reset)]):
    bax = fig.add_axes([0.30 + i * 0.10, 0.02, 0.09, 0.06])
    btn = Button(bax, text, color='lightgray', hovercolor='white')
    btn.on_clicked(handler)
    buttons.append(btn)

refresh()

timer = fig.canvas.new_timer(interval=TICK_MS)
timer.add_callback(tick)
timer.start()

plt.show()