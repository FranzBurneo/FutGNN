from pathlib import Path
import argparse
import re
import time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

APP = "https://vdotspain.shinyapps.io/Eventing2csv/"


def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\r\n\t]+', " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def debug_dump(page, dbg_dir: Path, name: str):
    dbg_dir.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(dbg_dir / f"{name}.png"), full_page=True)
    except Exception:
        pass
    try:
        (dbg_dir / f"{name}.html").write_text(page.content(), encoding="utf-8", errors="ignore")
    except Exception:
        pass


def close_info_modal(page) -> bool:
    """
    Cierra el modal tipo SweetAlert2 (swal2) que aparece al entrar:
    "Información Importante" + botón OK.
    """
    # SweetAlert2 clásico: .swal2-confirm
    try:
        btn = page.locator("button.swal2-confirm, .swal2-confirm").first
        if btn.count() > 0 and btn.is_visible():
            btn.click(timeout=2500, force=True)
            page.wait_for_timeout(350)
            return True
    except Exception:
        pass

    # Fallback por rol + texto
    try:
        b2 = page.get_by_role("button", name=re.compile(r"^ok$", re.I))
        if b2.count() > 0 and b2.first.is_visible():
            b2.first.click(timeout=2500, force=True)
            page.wait_for_timeout(350)
            return True
    except Exception:
        pass

    return False


def click_if_exists(page, patterns, timeout_ms=1500) -> bool:
    """Intenta clickear un botón/link cuyo texto matchee alguno de los patterns."""
    for pat in patterns:
        # Botón
        try:
            b = page.get_by_role("button", name=pat)
            if b.count() > 0 and b.first.is_visible():
                b.first.click(timeout=timeout_ms, force=True)
                return True
        except Exception:
            pass
        # Link
        try:
            a = page.get_by_role("link", name=pat)
            if a.count() > 0 and a.first.is_visible():
                a.first.click(timeout=timeout_ms, force=True)
                return True
        except Exception:
            pass
    return False


def wait_download_control(page, timeout_ms=180000):
    """
    Espera a que aparezca un control que dispare descarga:
    - a.shiny-download-link (muy común en Shiny)
    - a[download]
    - cualquier link/botón con texto Download/Descargar/CSV
    """
    candidates = page.locator(
        "a.shiny-download-link, "
        "a[download], "
        "a:has-text('Download'), a:has-text('Descargar'), a:has-text('CSV'), "
        "button:has-text('Download'), button:has-text('Descargar'), button:has-text('CSV')"
    )

    # Debe aparecer algo visible
    candidates.first.wait_for(state="visible", timeout=timeout_ms)

    # Y debe ser usable (si es <a>, debe tener href real)
    page.wait_for_function(
        """() => {
          const el =
            document.querySelector('a.shiny-download-link') ||
            document.querySelector('a[download]') ||
            Array.from(document.querySelectorAll('a')).find(x => /download|descargar|csv/i.test(x.textContent || '')) ||
            Array.from(document.querySelectorAll('button')).find(x => /download|descargar|csv/i.test(x.textContent || ''));

          if (!el) return false;

          if (el.tagName.toLowerCase() === 'a') {
            const href = el.getAttribute('href');
            if (!href || href === '#') return false;
          }

          const disabled = el.getAttribute('disabled') !== null || el.classList.contains('disabled');
          return !disabled;
        }""",
        timeout=timeout_ms
    )

    return candidates.first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Carpeta raíz con HTML (recursivo)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Carpeta salida para CSV")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--only-valid", action="store_true", help="Omitir HTML sin matchCentreData")
    ap.add_argument("--timeout", type=int, default=180000, help="Timeout ms para procesar/descargar")
    ap.add_argument("--limit", type=int, default=0, help="Procesa solo N archivos (0 = todos)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    dbg_dir = out_dir / "_debug_eventing2csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    htmls = sorted(list(in_dir.rglob("*.htm")) + list(in_dir.rglob("*.html")))
    if args.limit and args.limit > 0:
        htmls = htmls[: args.limit]

    print(f"HTML encontrados: {len(htmls)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()

        for f in htmls:
            # Validación opcional: evita 502 y páginas incompletas
            if args.only_valid:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                if "matchCentreData" not in txt:
                    print(f"[SKIP] sin matchCentreData: {f.name}")
                    continue

            target_csv = out_dir / (safe_name(f.stem) + ".csv")
            if target_csv.exists():
                print(f"[SKIP] ya existe: {target_csv.name}")
                continue

            print(f"\n-> Subiendo: {f.name}")

            try:
                page.goto(APP, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(1500)
                close_info_modal(page)

                # Asegura pestaña WhoScored si está
                click_if_exists(page, [re.compile("whoscored", re.I)], timeout_ms=2000)
                page.wait_for_timeout(700)
                close_info_modal(page)

                # Upload
                file_inputs = page.locator('input[type="file"]')
                if file_inputs.count() == 0:
                    debug_dump(page, dbg_dir, "no_file_input")
                    raise RuntimeError("No encontré input[type=file]. Posible cambio de UI en la app.")

                file_inputs.nth(0).set_input_files(str(f))
                page.wait_for_timeout(900)
                close_info_modal(page)

                # A veces requiere convertir/generar antes de habilitar descarga
                clicked_convert = click_if_exists(
                    page,
                    [
                        re.compile(r"(convert|generar|generate|procesar|process|run|ejecutar)", re.I),
                        re.compile(r"(submit|start)", re.I),
                    ],
                    timeout_ms=2500
                )
                if clicked_convert:
                    page.wait_for_timeout(800)
                    close_info_modal(page)

                # Espera el control de descarga
                close_info_modal(page)
                dl_control = wait_download_control(page, timeout_ms=args.timeout)

                # Descargar
                with page.expect_download(timeout=args.timeout) as dl_info:
                    dl_control.click(timeout=30000, force=True)

                dl = dl_info.value
                dl.save_as(str(target_csv))
                print(f"[OK] {target_csv.name}")

                # Respiro para no saturar el servicio
                time.sleep(0.6)

            except PWTimeoutError as e:
                print(f"[TIMEOUT] {f.name}: {e}")
                debug_dump(page, dbg_dir, safe_name(f.stem)[:80] + "_timeout")
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")
                debug_dump(page, dbg_dir, safe_name(f.stem)[:80] + "_error")

        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()