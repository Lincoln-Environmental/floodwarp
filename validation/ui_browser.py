from pathlib import Path
import json
import base64
import time
from playwright.sync_api import sync_playwright


def wait_text(*, page, selector, expected, timeout_s=30., contains=False):
    deadline = time.monotonic()+timeout_s
    while time.monotonic() < deadline:
        actual = page.locator(selector).text_content()
        if expected in actual if contains else actual == expected:
            return
        page.wait_for_timeout(timeout=100)
    raise RuntimeError(f'Timed out waiting for {selector}: {actual!r}')


def exercise_interface(*, url, source, output, screenshot):
    with sync_playwright() as runtime:
        browser = runtime.firefox.launch(headless=True)
        page = browser.new_page(viewport={'width': 1250, 'height': 1100})
        errors = []
        def record_error(error):
            errors.append(str(error))

        page.on(event='pageerror', f=record_error)
        page.goto(url=url)
        deadline = time.monotonic()+30.
        while page.locator('#controls').is_disabled():
            if time.monotonic() > deadline:
                raise RuntimeError('Interface initialization did not complete')
            page.wait_for_timeout(timeout=100)
        page.locator('#example').click(force=True)
        page.locator('#dem').fill(value=str(source/'dem.tif'))
        page.locator('#rainfall_csv').fill(value=str(source/'rainfall.csv'))
        page.locator('#output_parent').fill(value=str(output))
        page.locator('#domain_mode').select_option(value='raster')
        page.locator('#domain').fill(value=str(source/'domain.tif'))
        page.locator('summary').click(force=True)
        page.locator('#device').select_option(value='cpu')
        page.locator('#interval_s').fill(value='30')
        page.locator('summary').click(force=True)
        page.locator('#validate').click(force=True)
        wait_text(page=page, selector='#message', expected='Inputs valid.')
        old_output = page.locator('#output').inner_text()
        page.locator('#run').click(force=True)
        wait_text(page=page, selector='#message', expected='Simulation started.')
        deadline = time.monotonic()+30.
        while page.locator('#output').inner_text() == old_output:
            if time.monotonic() > deadline:
                raise RuntimeError('Run did not create a distinct output directory')
            page.wait_for_timeout(timeout=100)
        wait_text(page=page, selector='#state', expected='Completed', timeout_s=60.)
        assert not errors, errors
        assert page.locator('#maximum-preview').is_visible()
        page.locator('#maximum-preview').wait_for(state='visible')
        actual_output = page.locator('#output').inner_text()
        assert (Path(actual_output)/'checkpoint.npz').is_file()
        page.screenshot(path=str(screenshot), full_page=True, timeout=10000)
        page.reload()
        wait_text(page=page, selector='#state', expected='Completed')
        assert page.locator('#output').inner_text() == actual_output
        assert page.locator('#summary').inner_text().startswith('FV1')
        # Exercise the path selector and a real validation error through the page.
        page.locator('[data-browse="dem"]').click(force=True)
        page.wait_for_selector(selector='#entries button')
        page.locator('#close-picker').click(force=True)
        page.locator('#dem').fill(value='/nonexistent/floodwarp-dem.tif')
        page.locator('#rainfall_csv').fill(value=str(source/'rainfall.csv'))
        page.locator('#validate').click(force=True)
        wait_text(page=page, selector='#message', expected='No such file', contains=True)
        browser.close()
    return dict(interface='Firefox/Playwright; actual Validate and Run clicks', output=actual_output,
        completed=True, final_time_s=120., refresh_did_not_restart=True,
        file_selector=True, validation_error=True, page_errors=errors)


if __name__ == "__main__":
    url = 'http://127.0.0.1:8765'
    import floodwarp
    source = Path(floodwarp.__file__).parent/'ui_assets/example'
    output = Path('outputs/ui-browser').resolve()
    screenshot = Path('docs/ui.png').resolve()
    record = Path('validation/release_v0.1.2/browser.json').resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = exercise_interface(url=url, source=source, output=output, screenshot=screenshot)
    record.write_text(json.dumps(result, indent=2)+'\n')
