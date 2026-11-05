"""Opt-in browser QA of the actual completed interactive viewer."""
from pathlib import Path
import json
import os

from examples.orton_bradley.export_visualisation import atomic_json


def wait_for_time(*, page, seconds):
    page.wait_for_function(expression='(seconds) => window.floodwarpViewer?.inspect().time_s === seconds',
                           arg=seconds, timeout=120000)


def check_browser(*, url, executable, output):
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(executable), headless=True,
            args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-dev-shm-usage'])
        page = browser.new_page(viewport=dict(width=1500, height=1000), device_scale_factor=1)
        errors = []

        def record_error(error):
            errors.append(str(error))
            print(error, flush=True)

        def record_console(message):
            if message.type == 'error':
                record_error(error=message.text)

        page.on(event='console', f=record_console)
        page.on(event='pageerror', f=record_error)
        page.goto(url=url, wait_until='networkidle', timeout=120000)
        page.wait_for_function(expression="document.documentElement.dataset.viewerReady === 'true'", timeout=120000)
        initial = page.evaluate(expression='window.floodwarpViewer.inspect()')
        print(json.dumps(initial, indent=2), flush=True)
        assert initial['time_s'] == 88200. and initial['wet_triangles'] > 0
        page.screenshot(path=str(output / 'interactive_peak.png'))
        page.locator(selector='#start').click()
        wait_for_time(page=page, seconds=0.)
        dry = page.evaluate(expression='window.floodwarpViewer.inspect()')
        assert dry['wet_triangles'] == 0
        page.locator(selector='#end').click()
        wait_for_time(page=page, seconds=259200.)
        end = page.evaluate(expression='window.floodwarpViewer.inspect()')
        assert end['wet_triangles'] > 0
        page.locator(selector='#discharge-peak').click()
        wait_for_time(page=page, seconds=88200.)
        chart = page.locator(selector='#hydrograph')
        box = chart.bounding_box()
        chart.click(position=dict(x=34. + (box['width'] - 43.) / 2., y=box['height'] / 2.))
        wait_for_time(page=page, seconds=129600.)
        assert page.locator(selector='#clock').inner_text() == '36:00'
        chart.press(key='ArrowRight')
        wait_for_time(page=page, seconds=129900.)
        rain = page.locator(selector='#hyetograph')
        box = rain.bounding_box()
        rain.click(position=dict(x=34. + (box['width'] - 43.) / 3., y=box['height'] / 2.))
        wait_for_time(page=page, seconds=86400.)
        scene = page.locator(selector='#scene')
        bounds = scene.bounding_box()
        before = page.evaluate(expression='window.floodwarpViewer.inspect().camera')
        page.mouse.move(x=bounds['x'] + bounds['width'] * .5, y=bounds['y'] + bounds['height'] * .5)
        page.mouse.down()
        page.mouse.move(x=bounds['x'] + bounds['width'] * .65, y=bounds['y'] + bounds['height'] * .52, steps=12)
        page.mouse.up()
        after = page.evaluate(expression='window.floodwarpViewer.inspect().camera')
        assert before != after
        page.locator(selector='#zoom-in').click()
        zoomed = page.evaluate(expression='window.floodwarpViewer.inspect().camera')
        assert zoomed != after
        page.mouse.move(x=bounds['x'] + bounds['width'] * .5, y=bounds['y'] + bounds['height'] * .5)
        page.mouse.wheel(delta_x=0, delta_y=-250)
        page.wait_for_timeout(timeout=300)
        assert page.evaluate(expression='window.floodwarpViewer.inspect().camera') != zoomed
        target_before = page.evaluate(expression='window.floodwarpViewer.inspect().target')
        page.mouse.down(button='right')
        page.mouse.move(x=bounds['x'] + bounds['width'] * .7, y=bounds['y'] + bounds['height'] * .57, steps=8)
        page.mouse.up(button='right')
        assert page.evaluate(expression='window.floodwarpViewer.inspect().target') != target_before
        page.locator(selector='#reset').click()
        page.locator(selector='#inundation-peak').click()
        wait_for_time(page=page, seconds=86700.)
        page.screenshot(path=str(output / 'interactive_peak_extent.png'))
        page.locator(selector='#play').click()
        page.wait_for_function(expression='window.floodwarpViewer.inspect().time_s > 86700', timeout=120000)
        page.locator(selector='#play').click()
        # Rapid selections must commit only the final requested frame.
        page.evaluate(expression="() => {const s=document.getElementById('time');for(const i of [10,200,500,100]){s.value=i;s.dispatchEvent(new Event('input'));}}")
        wait_for_time(page=page, seconds=30000.)
        page.evaluate(expression="() => {const s=document.getElementById('time');for(const i of [200,100]){s.value=i;s.dispatchEvent(new Event('input'));}}")
        page.wait_for_timeout(timeout=1000)
        assert page.evaluate(expression='window.floodwarpViewer.inspect().time_s') == 30000.
        assert page.evaluate(expression='window.floodwarpViewer.inspect().cached_frames') <= 6
        # Drag on the hydrograph and check the final requested sample.
        box = chart.bounding_box()
        page.mouse.move(x=box['x'] + 34. + (box['width'] - 43.) * .25, y=box['y'] + box['height'] * .5)
        page.mouse.down()
        page.mouse.move(x=box['x'] + 34. + (box['width'] - 43.) * .75, y=box['y'] + box['height'] * .5, steps=8)
        page.mouse.up()
        wait_for_time(page=page, seconds=194400.)
        page.locator(selector='#discharge-peak').click()
        wait_for_time(page=page, seconds=88200.)
        page.set_viewport_size(viewport_size=dict(width=390, height=844))
        page.screenshot(path=str(output / 'interactive_mobile.png'), full_page=True)
        assert page.evaluate(expression='document.documentElement.scrollWidth <= innerWidth')
        fallback_page = browser.new_page(viewport=dict(width=1200, height=900))
        fallback_page.on(event='pageerror', f=record_error)
        fallback_page.on(event='console', f=record_console)
        fallback_page.add_init_script(script="const originalExtension=WebGL2RenderingContext.prototype.getExtension;WebGL2RenderingContext.prototype.getExtension=function(name){return name==='EXT_clip_control'?null:originalExtension.call(this,name);};")
        fallback_page.goto(url=url, wait_until='networkidle', timeout=120000)
        fallback_page.wait_for_function(expression="document.documentElement.dataset.viewerReady === 'true'", timeout=120000)
        fallback = fallback_page.evaluate(expression='window.floodwarpViewer.inspect()')
        assert fallback['depth_buffer'] == 'logarithmic' and fallback['wet_triangles'] > 0
        fallback_page.screenshot(path=str(output / 'interactive_logarithmic_fallback.png'))
        assert not errors, errors
        report = dict(status='passed', initial=initial, dry=dry, final=end, fallback=fallback,
            checks=['WebGL2 native terrain/water', 'dry and final fields', 'hydrograph time selection',
                'rainfall time selection', 'keyboard stepping', 'orbit', 'zoom', 'reset',
                'playback', 'rapid-selection race', 'return-to-current-frame race', 'chart dragging', 'wheel zoom', 'pan', 'bounded cache', 'mobile layout', 'logarithmic depth fallback'], page_errors=errors)
        atomic_json(path=output / 'browser_qa.json', value=report)
        browser.close()
    return report


def check_data(*, dataset, browser_root, output):
    """Independently compare every browser payload to the original FP64 result."""
    import gzip
    import hashlib
    import numpy as np
    from examples.orton_bradley.reproduce import file_hash

    manifest = json.loads((browser_root / 'data/manifest.json').read_text())
    with np.load(file=dataset / 'metadata/terrain.npz', allow_pickle=False) as source:
        dem, active, valid = source['dem'], source['active'], source['valid_dem']
    count = dem.size
    terrain = gzip.decompress(data=(browser_root / manifest['terrain']['path']).read_bytes())
    if hashlib.sha256(terrain).hexdigest() != manifest['terrain']['raw_sha256']:
        raise ValueError('Terrain payload checksum differs')
    xyz = np.frombuffer(buffer=terrain, dtype='<f4', count=count * 3).reshape(-1, 3)
    np.testing.assert_array_equal(actual=np.frombuffer(buffer=terrain, dtype=np.uint8,
        offset=count * 15, count=count).reshape(active.shape).astype(bool), desired=valid)
    np.testing.assert_array_equal(actual=np.frombuffer(buffer=terrain, dtype=np.uint8,
        offset=count * 16, count=count).reshape(active.shape).astype(bool), desired=active)
    display_dem = xyz[:, 1].reshape(dem.shape)
    maximum_depth_error = maximum_height_error = 0.
    times = []
    for index, record in enumerate(manifest['frames']):
        source_path = dataset / 'frames' / f'depth-{int(record["time_s"]):06d}.npz'
        if file_hash(path=source_path) != record['source_sha256']:
            raise ValueError('Native source changed')
        binary = gzip.decompress(data=(browser_root / record['path']).read_bytes())
        if hashlib.sha256(binary).hexdigest() != record['raw_sha256']:
            raise ValueError('Browser frame checksum differs')
        display = np.frombuffer(buffer=binary, dtype='<f4', count=count).reshape(active.shape)
        wet = np.frombuffer(buffer=binary, dtype=np.uint8, offset=count * 4).reshape(active.shape)
        with np.load(file=source_path, allow_pickle=False) as source:
            depth = source['depth']
            if float(source['time_s']) != record['time_s']:
                raise ValueError('Frame time differs')
        expected_wet = active & (depth >= manifest['configuration']['wet_threshold_m'])
        np.testing.assert_array_equal(actual=wet.astype(bool), desired=expected_wet)
        np.testing.assert_array_equal(actual=display, desired=depth.astype('<f4'))
        error = float(np.max(np.abs(display.astype(np.float64) - depth)))
        height_error = float(np.max(np.abs((display_dem + display).astype(np.float64)[active] -
                                            (dem + depth)[active])))
        if error > np.finfo(np.float32).eps * max(1., float(depth.max())):
            raise ArithmeticError('Display depth error exceeds Float32 conversion precision')
        height_bound = np.finfo(np.float32).eps * max(1., float(np.max(np.abs(dem[active]))) + float(depth.max()))
        if height_error > height_bound:
            raise ArithmeticError('Display water height exceeds Float32 rounding bound')
        maximum_depth_error = max(maximum_depth_error, error)
        maximum_height_error = max(maximum_height_error, height_error)
        times.append(record['time_s'])
        if index % 120 == 0:
            print(f'Verified browser payload {index + 1}/{len(manifest["frames"])}', flush=True)
    np.testing.assert_array_equal(actual=times, desired=np.arange(0., 259201., 300.))
    report = dict(status='passed', frames=len(times), native_grid=True,
        maximum_display_depth_error_m=maximum_depth_error,
        maximum_display_water_elevation_error_m=maximum_height_error,
        wet_classification='Exact match to original FP64 values before display conversion',
        crs=manifest['crs'], vertical_datum=manifest['vertical_datum'],
        precision='Float32 display only; original FP64 simulation and files unchanged')
    atomic_json(path=output / 'data_qa.json', value=report)
    return report


if __name__ == "__main__":
    url = 'http://127.0.0.1:8766/'
    executable = Path('/home/patrickdurney/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome')
    output = Path('outputs/orton_bradley_interactive/qa')
    dataset = Path('outputs/orton_bradley_visualisation')
    browser_root = Path('outputs/orton_bradley_interactive')
    action = os.environ.get('FLOODWARP_INTERACTIVE_CHECK', 'all')
    if action in ('data', 'all'):
        print(json.dumps(check_data(dataset=dataset, browser_root=browser_root, output=output), indent=2))
    if action in ('browser', 'all'):
        print(json.dumps(check_browser(url=url, executable=executable, output=output), indent=2))
    if action not in ('data', 'browser', 'all'):
        raise ValueError('Check action must be data, browser or all')
