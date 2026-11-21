"""Start the Streamlit demo, classify a bundled sample image and save docs/demo.png (needs playwright + Edge).

Usage: python scripts/screenshot_demo.py data/dataset-resized/glass/glass10.jpg docs/demo.png
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main(image, out, port=8765):
    server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', str(port),
                               '--server.headless', 'true', '--browser.gatherUsageStats', 'false'], cwd=ROOT)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f'http://localhost:{port}/_stcore/health', timeout=2)
                break
            except OSError:
                time.sleep(2)
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context('D:/work/_cache/edge-aiml', channel='msedge', headless=True,
                                                           viewport={'width': 1280, 'height': 1500})
            page = browser.new_page()
            page.goto(f'http://localhost:{port}', wait_until='domcontentloaded')
            page.wait_for_selector('input[type=file]', state='attached', timeout=90000)
            page.set_input_files('input[type=file]', str(image))
            page.get_by_role('button', name='Classify image').click()
            page.get_by_text('Where the model looked').wait_for(timeout=120000)
            time.sleep(2)
            page.screenshot(path=str(out), full_page=True)
            browser.close()
    finally:
        server.terminate()


if __name__ == '__main__':
    main(*sys.argv[1:3])
