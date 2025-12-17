from pathlib import Path
import re
from playwright.sync_api import sync_playwright

APP = "https://vdotspain.shinyapps.io/Eventing2csv/"

def main(in_dir: str, out_dir: str):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    htmls = sorted([*in_dir.glob("*.htm"), *in_dir.glob("*.html")])
    if not htmls:
        raise SystemExit("No hay HTMLs en la carpeta")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        page.goto(APP, wait_until="domcontentloaded")

        # Espera a que Shiny pinte inputs (puede tardar)
        page.wait_for_timeout(6000)

        for hp in htmls:
            page.goto(APP, wait_until="domcontentloaded")
            page.wait_for_timeout(6000)

            # Selecciona pestaña WhoScored (por si acaso)
            try:
                page.get_by_role("link", name=re.compile("Whoscored", re.I)).click(timeout=2000)
            except:
                pass

            # Encuentra el input de archivos y carga el HTML
            file_inputs = page.locator('input[type="file"]')
            if file_inputs.count() == 0:
                raise RuntimeError("No encontré input[type=file] en la página (cambió la UI).")
            file_inputs.nth(0).set_input_files(str(hp))

            # Busca un botón de descarga que contenga "csv"/"descarg"
            btn = page.get_by_role("button", name=re.compile(r"(csv|descarg)", re.I)).first

            with page.expect_download() as dl_info:
                btn.click()

            dl = dl_info.value
            target = out_dir / f"{hp.stem}.csv"
            dl.save_as(str(target))
            print(f"OK -> {target.name}")

        ctx.close()
        browser.close()

if __name__ == "__main__":
    # python auto_eventing2csv.py ./htmls ./csvs
    import sys
    if len(sys.argv) != 3:
        print("Uso: python auto_eventing2csv.py <carpeta_htmls> <carpeta_csvs>")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])