"""Checks of the bandwidth, hosts and power study (core/studies/bandwidth.py) against independent references.

- bandwidth.stated_mode_rates: a stream's data rate (width × height × fps × bits per pixel) against the data rate a
  maker states for that mode. The only maker in the library that states one is ams OSRAM for the NanEyeC: "75 Mbps
  at 58 Hz" over its single-ended SEIM interface and "63 Mbps at 49 Hz" over LVDS, for its 320 × 320 RAW10 image
  (datasheet, as quoted in the library entry's note). Those are the interface's serial rates, which carry the pixels
  and the line and frame framing, so the pixel payload must be below them, and the share the framing takes is
  reported. The figures are typed in here from the entry's note, and the stream is read from the entry's JSON file,
  not through the study.
- bandwidth.spec_figures: the capacities and the bus-power allowances the study uses against the figures the USB
  specifications state in other places: USB 2.0 Table 5-10's "Max Bandwidth" 53 248 000 bytes/s (the study derives
  it from 13 transactions × 512 bytes × 8000 microframes); USB 3.2 §4.4.11's 450 MB/s and 1.1 GB/s; USB 3.2 Table
  3-1's "high (900 mA)" and "high (1,500 mA)" and USB 2.0's "high (500 mA)" (the study derives them from unit loads
  × their count, USB 2.0 §7.2.1 and USB 3.2 §11.4.1).
- bandwidth.budget_recompute: the whole budget on seeded random topologies (hosts, controllers of each kind, powered
  and bus-powered hubs, sensors of eleven library devices with random stream modes) against a re-computation written
  out here that shares nothing with the study but the library's JSON files: its own stream lookup and arithmetic,
  its own walk up the hub tree, its own table of each device's USB generation and the USB rules typed in again.
"""
from __future__ import annotations

import glob
import json
import os


from .suite import Outcome, check, figure, series

# the NanEyeC's stated serial rates (library entry note, from its datasheet): (fps, Mbit/s, interface)
NANEYEC_STATED = ((58.0, 75.0, 'SEIM'), (49.0, 63.0, 'LVDS'))
# typed in again for the re-computation (not imported from the study)
CAP = {'usb2': 53_248_000 * 8 / 1e6, 'gen1': 450 * 8.0, 'gen2': 1100 * 8.0, 'gen2x2': 2 * 1100 * 8.0}
RANK = {'usb2': 0, 'gen1': 1, 'gen2': 2, 'gen2x2': 3}
OVERHEAD = 0.05
HIGH_MA = {'usb2': 500.0, 'usb3': 900.0, 'usb3x2': 1500.0}
DEVICE_GEN = {                                    # read off each entry's "interface" by hand
    'intel-realsense-d405': 'gen1', 'intel-realsense-d415': 'gen1', 'intel-realsense-d435': 'gen1',
    'intel-realsense-d455': 'gen1', 'e-con-see3cam-20cug': 'gen1', 'e-con-see3cam-24cug': 'gen1',
    'stereolabs-zed-2i': 'gen1', 'orbbec-gemini-2': 'gen1', 'orbbec-femto-bolt': 'gen1',
    'intel-realsense-d457': None, 'stereolabs-zed-x': None,
}
GMSL_DEVICES = ('intel-realsense-d457', 'stereolabs-zed-x')
HOST_GEN = {'usb-3.2-gen1-host': 'gen1', 'usb-3.2-gen2-host': 'gen2', 'usb-3.2-gen2x2-host': 'gen2x2', '': 'gen1'}


def _library_json() -> dict:
    from ..core import library as LB
    out = {}
    for f in glob.glob(os.path.join(LB.BUILTIN_DIR, '**', '*.json'), recursive=True):
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        out[d['id']] = d
    return out


@check('bandwidth.stated_mode_rates', 'bandwidth', 'Stream data rate against the maker\'s stated rate for the mode',
       'how far the pixel payload (width × height × fps × bits per pixel) of the NanEyeC\'s 320 × 320 RAW10 mode '
       'exceeds the serial rate the maker states for it (0 when below; it must also be at least half of it)',
       'ams OSRAM NanEyeC datasheet: 75 Mbps at 58 Hz (SEIM), 63 Mbps at 49 Hz (LVDS)', 0.0, 'fraction',
       refs=('naneyec',))
def _stated_rates(ctx):
    from ..core import library as LB
    from ..core.project import Project, Sensor
    from ..core.studies import bandwidth as BW
    d = _library_json()['ams-osram-naneyec']
    s = d['streams'][0]
    st = BW.settings_of(Project())
    sensor = Sensor(name='NanEyeC', kind='camera', model_id='ams-osram-naneyec')
    entry = LB.get('ams-osram-naneyec')
    worst, rows, ratios, ok = 0.0, [], [], True
    bus = BW.device_bus(entry)
    default_ov = BW.DEFAULT_OVERHEAD[bus if bus in BW.DEFAULT_OVERHEAD else 'other']
    for fps, stated, iface in NANEYEC_STATED:
        got = BW.sensor_streams(sensor, entry, [BW.stream_id(s, fps)], st)['payload_mbps']
        ref_payload = s['width'] * s['height'] * fps * s['bits_per_pixel'] / 1e6
        ratio = got / stated
        worst = max(worst, ratio - 1.0, abs(got - ref_payload) / ref_payload)
        ok = ok and ratio >= 0.5
        ratios.append(ratio)
        wire = got * (1.0 + default_ov)
        rows.append({'interface': iface, 'fps': fps, 'stated_mbps': stated, 'payload_mbps': got,
                     'payload_share_of_stated': ratio, 'implied_framing_overhead': stated / got - 1.0,
                     'study_default_overhead': default_ov, 'study_wire_mbps': wire,
                     'study_wire_below_stated': 1.0 - wire / stated})
    lo = min(r['implied_framing_overhead'] for r in rows) * 100
    hi = max(r['implied_framing_overhead'] for r in rows) * 100
    short = max(r['study_wire_below_stated'] for r in rows) * 100
    return Outcome(max(0.0, worst), {'modes': rows}, ok=ok,
                   figure=figure('NanEyeC: pixel payload against the stated serial rate', 'fps', 'Mbit/s',
                                 [series('stated', [r['fps'] for r in rows], [r['stated_mbps'] for r in rows],
                                         'points'),
                                  series('payload (model)', [r['fps'] for r in rows],
                                         [r['payload_mbps'] for r in rows], 'points'),
                                  series(f'on the wire (model, default overhead {default_ov * 100:g} %)',
                                         [r['fps'] for r in rows], [r['study_wire_mbps'] for r in rows], 'points')]),
                   note='The stated rates are the interface\'s serial bit rates, which carry the line and frame '
                        'framing as well as the pixels: the pixel payload must not exceed them (tolerance 0) and the '
                        f'implied framing share, {lo:.1f} to {hi:.1f} %, is reported, not judged. It disagrees with '
                        f'the study\'s default overhead ({default_ov * 100:g} %): the study\'s on-the-wire figure for '
                        f'these modes is up to {short:.1f} % below the stated serial rate. The default is meant for '
                        'what a USB or GMSL link adds on top of the pixels, not for this sensor\'s own serial framing; '
                        'for a link that carries the NanEyeC\'s raw serial stream, enter the stated rate as a measured '
                        'figure or an overhead of about this share. No other maker in the library states a data rate '
                        'for a stream mode, so no USB or GMSL device can be checked this way.')


@check('bandwidth.spec_figures', 'bandwidth', 'USB capacities and bus-power allowances against the specifications',
       'largest relative difference between the study\'s derived capacities (USB 2.0 bulk, USB 3.2 Gen 1, Gen 2, '
       'Gen 2x2) and bus-power allowances (USB 2.0, USB 3 one and two lanes) and the figures the specifications state '
       'elsewhere', 'USB 2.0 Table 5-10 and Table 3-1 of USB 3.2; USB 3.2 §4.4.11', 1e-12, 'relative',
       refs=('usb20_spec', 'usb32_spec'))
def _spec(ctx):
    from ..core.studies import bandwidth as BW
    pairs = [
        ('USB 2.0 bulk, bytes/s (Table 5-10)', BW.CAPACITY_MBPS['usb2'] * 1e6 / 8, 53_248_000.0),
        ('USB 3.2 Gen 1x1 effective, MB/s (§4.4.11)', BW.CAPACITY_MBPS['gen1'] / 8, 450.0),
        ('USB 3.2 Gen 2x1 effective, MB/s (§4.4.11)', BW.CAPACITY_MBPS['gen2'] / 8, 1100.0),
        ('USB 3.2 Gen 2x2 effective, MB/s (doubled, §4.4.11)', BW.CAPACITY_MBPS['gen2x2'] / 8, 2200.0),
        ('USB 2.0 high-power, mA (USB 3.2 Table 3-1)', BW.UNIT_LOAD_MA['usb2'] * BW.HIGH_POWER_UNITS['usb2'], 500.0),
        ('USB 3 one lane high-power, mA (Table 3-1)', BW.UNIT_LOAD_MA['usb3'] * BW.HIGH_POWER_UNITS['usb3'], 900.0),
        ('USB 3 two lanes high-power, mA (Table 3-1)', BW.UNIT_LOAD_MA['usb3x2'] * BW.HIGH_POWER_UNITS['usb3x2'],
         1500.0),
    ]
    errs = [abs(a - b) / b for _n, a, b in pairs]
    return Outcome(float(max(errs)), {'figures': [{'what': n, 'study': a, 'stated': b} for n, a, b in pairs]},
                   note='Each figure is derived in the study from other statements of the same specification (unit '
                        'loads and their count, transactions per microframe); the tolerance is rounding only.')


# ====================================================================== random topologies
POOL = list(DEVICE_GEN)


def _random_project(rng, lib):
    from ..core.project import Project, Sensor, new_uid
    p = Project('Bandwidth check rig')
    hosts, ctrls, hubs, links = [], [], [], []
    for h in range(int(rng.integers(1, 3))):
        lid = str(rng.choice(['', 'usb-3.2-gen1-host', 'usb-3.2-gen2-host']))
        cs = []
        for _k in range(int(rng.integers(1, 4))):
            kind = str(rng.choice(['usb3', 'usb3', 'usb2', 'gmsl']))
            prac = float(rng.uniform(500, 6000)) if rng.random() < 0.3 else None
            c = {'uid': new_uid(), 'name': f'C{len(ctrls)}', 'kind': kind, 'practical_mbps': prac,
                 'source': 'check' if prac else ''}
            cs.append(c)
            ctrls.append((c, lid))
        hosts.append({'uid': new_uid(), 'name': f'H{h}', 'library_id': lid, 'controllers': cs})
    usb_ctrls = [c for c, _l in ctrls if c['kind'] in ('usb3', 'usb2')]
    for _k in range(int(rng.integers(0, 4))):
        if not usb_ctrls:
            break
        up = [c['uid'] for c in usb_ctrls] + [b['uid'] for b in hubs]
        hubs.append({'uid': new_uid(), 'name': f'Hub {len(hubs)}', 'powered': bool(rng.random() < 0.5),
                     'upstream': str(rng.choice(up))})
    for k in range(int(rng.integers(2, 8))):
        mid = str(rng.choice(POOL))
        d = lib[mid]
        s = Sensor(name=f'S{k}', kind='depth' if d['kind'] == 'depth' else 'camera', model_id=mid, slot=k)
        gm = mid in GMSL_DEVICES
        targets = ([c['uid'] for c, _l in ctrls if c['kind'] == 'gmsl'] if gm else
                   [c['uid'] for c in usb_ctrls] + [b['uid'] for b in hubs])
        if not targets:
            continue                                  # every sensor of the check rig has a link (explicit streams)
        p.variant.sensors.append(s)
        streams = [x for x in d['streams'] if x.get('bits_per_pixel')]
        pick = rng.choice(len(streams), size=min(len(streams), int(rng.integers(1, 4))), replace=False)
        ids = []
        for i in pick:
            x = streams[int(i)]
            fps = float(rng.choice(x['fps']))
            ids.append(f'{x["imager"]}|{x["width"]}x{x["height"]}|{fps:g}|{x.get("format") or ""}')
        links.append({'sensor_uid': s.uid, 'stream_ids': ids, 'via': str(rng.choice(targets)), 'cable_uid': ''})
    p.rig = {'hosts': hosts, 'hubs': hubs, 'links': links}
    return p


def _recompute(p, lib):
    """The budget written out again (module docstring)."""
    ctrl = {c['uid']: (c, h['library_id']) for h in p.rig['hosts'] for c in h['controllers']}
    hubs = {b['uid']: b for b in p.rig['hubs']}

    def chain(uid):
        path = []
        while uid not in ctrl:
            path.append(uid)
            uid = hubs[uid]['upstream']
        return uid, path

    def gen_of(cu):
        c, lid = ctrl[cu]
        return 'usb2' if c['kind'] == 'usb2' else HOST_GEN[lid] if c['kind'] == 'usb3' else None

    sens = {s.uid: s for s in p.variant.sensors}
    out = {'wire': {}, 'link_load': {}, 'ctrl_load': {u: 0.0 for u in ctrl}, 'draw': {}, 'allow': {},
           'hub_draw': {}}
    payload_total = 0.0
    for k in p.rig['links']:
        s = sens[k['sensor_uid']]
        d = lib[s.model_id]
        pay = 0.0
        for sid in k['stream_ids']:
            im, wh, fps, fmt = sid.split('|')
            w, h = (int(v) for v in wh.split('x'))
            x = next(x for x in d['streams'] if x['imager'] == im and x['width'] == w and x['height'] == h
                     and (x.get('format') or '') == fmt)
            pay += w * h * float(fps) * x['bits_per_pixel'] / 1e6
        payload_total += pay
        wire = pay * (1 + OVERHEAD)
        out['wire'][s.uid] = wire
        cu, path = chain(k['via'])
        out['ctrl_load'][cu] += wire
        c, _lid = ctrl[cu]
        cg = gen_of(cu)
        if cg is not None:
            dg = DEVICE_GEN[s.model_id]
            g = cg if dg is None else min(dg, cg, key=RANK.get)
            cap = CAP[g]
            if c['practical_mbps'] and c['practical_mbps'] < cap:
                cap = c['practical_mbps']
            out['link_load'][s.uid] = wire / cap
            # bus power at the nominal 5 V
            draw = d['power_w'] / 5.0 * 1000.0
            out['draw'][s.uid] = draw
            fl = 'usb2' if cg == 'usb2' else 'usb3'
            if k['via'] in ctrl or hubs[k['via']]['powered']:
                out['allow'][s.uid] = HIGH_MA[fl]
            elif fl == 'usb2':
                out['allow'][s.uid] = 100.0
    # bus-powered hubs draw what is below them; powered hubs draw nothing counted
    def hub_draw(bu):
        if hubs[bu]['powered']:
            return 0.0
        t = sum(out['draw'].get(k['sensor_uid'], 0.0) for k in p.rig['links'] if k['via'] == bu)
        t += sum(hub_draw(b2) for b2, hb in hubs.items() if hb['upstream'] == bu)
        return t
    for bu in hubs:
        cu, _ = chain(bu)
        if gen_of(cu) is not None:
            out['hub_draw'][bu] = hub_draw(bu) if not hubs[bu]['powered'] else 0.0
    mb_s = payload_total * 1.0 * 1.03 / 8.0
    out['gb_episode'] = mb_s * 60.0 / 1000.0
    out['gb_campaign'] = out['gb_episode'] * 100
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-9)


@check('bandwidth.budget_recompute', 'bandwidth', 'Bandwidth budget against an independent re-computation',
       'largest relative difference over seeded random topologies between the study and the re-computation: each '
       'sensor\'s rate on the wire and link load, each controller\'s load, each bus-powered device\'s current and '
       'port allowance, each bus-powered hub\'s draw, and the recording per episode and per campaign',
       'the budget written out again in the check from the library\'s JSON files (its own stream lookup, hub walk, '
       'device generations and USB rules)', 1e-9, 'relative', refs=('usb20_spec', 'usb32_spec'))
def _recompute_check(ctx):
    from ..core.studies import bandwidth as BW
    lib = _library_json()
    rng = ctx.rng('bandwidth.budget_recompute')
    n = 8 if ctx.quick else 40
    worst, compared, xs, ys = 0.0, 0, [], []
    for _t in range(n):
        p = _random_project(rng, lib)
        res = BW.budget(p)
        ref = _recompute(p, lib)
        rows = {r['uid']: r for r in res['sensors']}
        for u, w in ref['wire'].items():
            pairs = [(rows[u]['wire_mbps'], w)]
            if u in ref['link_load']:
                pairs.append((rows[u]['link_load'], ref['link_load'][u]))
            if u in ref['draw']:
                pairs.append((rows[u]['draw_ma'], ref['draw'][u]))
            if u in ref['allow']:
                pairs.append((rows[u]['allowance_ma'], ref['allow'][u]))
            for a, b in pairs:
                worst = max(worst, _rel(a, b) if a is not None else float('inf'))
                compared += 1
            xs.append(w)
            ys.append(rows[u]['wire_mbps'])
        for c in res['controllers']:
            worst = max(worst, _rel(c['load_mbps'], ref['ctrl_load'][c['uid']]) if ref['ctrl_load'][c['uid']] else
                        abs(c['load_mbps']))
            compared += 1
        for h in res['hubs']:
            if h['uid'] in ref['hub_draw'] and not h['powered']:
                worst = max(worst, _rel(h['draw_ma'], ref['hub_draw'][h['uid']]) if ref['hub_draw'][h['uid']] else
                            abs(h['draw_ma'] or 0.0))
                compared += 1
        for k in ('gb_episode', 'gb_campaign'):
            got = res['disk']['gb_per_episode' if k == 'gb_episode' else 'gb_per_campaign']
            worst = max(worst, _rel(got, ref[k]) if ref[k] else abs(got))
            compared += 1
    return Outcome(float(worst), {'topologies': n, 'figures_compared': compared,
                                  'devices': POOL},
                   figure=figure('Rate on the wire per sensor', 're-computation (Mbit/s)', 'study (Mbit/s)',
                                 [series('sensors', xs, ys, 'points')], diag=True),
                   note='Both paths do the same arithmetic in a different order on the same stated figures, so they '
                        'agree to rounding (1e-9 relative leaves room for summation order); a wrong stream lookup, a '
                        'hub counted on the wrong controller or a wrong allowance shows as a whole-number factor.')
