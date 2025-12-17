import argparse
import os
import re
import time
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

BASE = "https://es.whoscored.com"

MONTH_MAP = {
    # ES (3 letras) + EN (3 letras) por si el sitio mezcla abreviaturas
    "ene": 1, "jan": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4, "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8, "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12, "dec": 12,
}

import unicodedata

def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = (name or "").strip()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[<>:"|?*\r\n\t]', " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len] if len(name) > max_len else name

def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 2
    while True:
        cand = f"{base} ({k}){ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def ensure_abs_url(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return BASE + href
    return BASE + "/" + href

def match_id_from_url(url: str) -> str | None:
    m = re.search(r"/matches/(\d+)/", url)
    return m.group(1) if m else None

def get_month_label(page) -> str | None:
    """Lee el label visible tipo: 'may 2025'."""
    try:
        loc = page.locator("span.toggleDatePicker").first
        loc.wait_for(state="visible", timeout=15000)
        return (loc.inner_text() or "").strip()
    except Exception:
        return None

def label_to_ym(label: str) -> str | None:
    """
    Convierte 'may 2025' -> '2025-05'
    """
    if not label:
        return None
    s = label.strip().lower()
    parts = s.split()
    if len(parts) < 2:
        return None
    mon = parts[0][:3]
    year = parts[1]
    if mon not in MONTH_MAP:
        return None
    try:
        y = int(year)
    except ValueError:
        return None
    m = MONTH_MAP[mon]
    return f"{y:04d}-{m:02d}"

def click_privacy_accept(page) -> bool:
    """
    Intenta cerrar el modal de privacidad / cookies.
    En tu captura aparece 'Aceptar todo'.
    """
    patterns = [
        re.compile(r"aceptar\s+todo", re.I),
        re.compile(r"accept\s+all", re.I),
        re.compile(r"aceptar", re.I),
        re.compile(r"accept", re.I),
        re.compile(r"i\s*agree|agree", re.I),
        re.compile(r"ok", re.I),
    ]

    # 1) por rol (mejor)
    for pat in patterns:
        try:
            btn = page.get_by_role("button", name=pat)
            if btn.count() > 0:
                btn.first.click(timeout=3000, force=True)
                page.wait_for_timeout(300)
                return True
        except Exception:
            pass

    # 2) fallback por CSS/texto
    candidates = [
        "button:has-text('Aceptar todo')",
        "button:has-text('Accept all')",
        "button:has-text('Aceptar')",
        "button:has-text('Accept')",
        "button:has-text('I Agree')",
        "button:has-text('Agree')",
        "button:has-text('OK')",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                loc.click(timeout=3000, force=True)
                page.wait_for_timeout(300)
                return True
        except Exception:
            pass

    return False

def nuke_overlays(page):
    """
    Elimina overlays típicos (ads/webpush/cookies) que interceptan clicks.
    En tus logs se ve: div.a__sc-np32r2-0.fDtvqm intercepts pointer events
    y en tu HTML hay componentes webpush y modal de privacidad.
    """
    try:
        page.evaluate("""
        () => {
          const selectors = [
            'div.a__sc-np32r2-0.fDtvqm',
            '.webpush-swal2-container',
            '#webpush-notification-center-open',
            '[role="dialog"]',
            '.qc-cmp2-container',
            '#qc-cmp2-container',
            '.fc-consent-root',
            '.fc-dialog',
            '.tp-modal',
            '.tp-backdrop'
          ];
          for (const sel of selectors) {
            document.querySelectorAll(sel).forEach(el => el.remove());
          }

          // Intenta borrar cualquier overlay fullscreen con z-index muy alto
          document.querySelectorAll('body *').forEach(el => {
            const st = window.getComputedStyle(el);
            if (st.position === 'fixed') {
              const zi = parseInt(st.zIndex || '0', 10);
              if (zi > 1000) {
                const r = el.getBoundingClientRect();
                if (r.width > window.innerWidth * 0.7 && r.height > window.innerHeight * 0.7) {
                  el.remove();
                }
              }
            }
          });
        }
        """)
    except Exception:
        pass

    try:
        page.keyboard.press("Escape")
    except Exception:
        pass

def get_stats_links(page):
    """
    Saca links de stats de fixtures:
    - preferido: <a ...><img alt="stats"></a>
    - fallback: anchors con /matches/.../live
    """
    links = []
    # 1) según tu HTML: <a ...><img alt="stats"></a>
    try:
        loc = page.locator("a:has(img[alt='stats'])")
        for i in range(loc.count()):
            href = loc.nth(i).get_attribute("href")
            if href and "/matches/" in href and "/live" in href:
                links.append(ensure_abs_url(href))
    except Exception:
        pass

    # 2) fallback
    if not links:
        try:
            loc = page.locator("a[href*='/matches/'][href*='/live']")
            for i in range(loc.count()):
                href = loc.nth(i).get_attribute("href")
                if href:
                    links.append(ensure_abs_url(href))
        except Exception:
            pass

    # dedup
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def wait_month_label_change(page, old_label: str, timeout_ms: int = 30000):
    """
    Espera hasta que cambie el texto de 'span.toggleDatePicker' (ej: may 2025 -> apr 2025).
    OJO: En Playwright sync, arg es keyword-only.
    """
    page.wait_for_function(
        """(oldLabel) => {
            const el = document.querySelector('span.toggleDatePicker');
            if (!el) return false;
            const now = (el.textContent || '').trim();
            return now.length > 0 && now !== oldLabel;
        }""",
        arg=old_label,
        timeout=timeout_ms
    )

def debug_dump(page, out_dir: str, name: str):
    safe_mkdir(out_dir)
    try:
        page.screenshot(path=os.path.join(out_dir, f"{name}.png"), full_page=True)
    except Exception:
        pass
    try:
        with open(os.path.join(out_dir, f"{name}.html"), "w", encoding="utf-8") as f:
            f.write(page.content())
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--until", required=True, help="Mes límite inclusive YYYY-MM (ej: 2024-08).")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--slowmo", type=int, default=0, help="ms (útil en headful)")
    ap.add_argument("--max-prev-retries", type=int, default=5, help="Intentos de cambiar de mes hacia atrás")
    args = ap.parse_args()

    safe_mkdir(args.out)
    dbg_dir = os.path.join(args.out, "_debug")
    safe_mkdir(dbg_dir)

    processed_months = set()
    seen_match_ids = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.headful,
            slow_mo=args.slowmo,
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = browser.new_context(
            locale="es-ES",
            timezone_id="America/Guayaquil",
            viewport={"width": 1400, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

        page = context.new_page()
        page.goto(args.fixtures_url, wait_until="domcontentloaded", timeout=60000)

        # intenta cerrar modales de entrada
        for _ in range(3):
            click_privacy_accept(page)
            nuke_overlays(page)
            page.wait_for_timeout(400)

        # espera el calendario
        try:
            page.locator("#dayChangeBtn-prev").wait_for(state="attached", timeout=60000)
        except PWTimeoutError:
            debug_dump(page, dbg_dir, "fixtures_no_prev_btn")
            raise RuntimeError("No cargó el calendario (#dayChangeBtn-prev). Revisa _debug.")

        while True:
            # limpiar modales/overlays siempre
            click_privacy_accept(page)
            nuke_overlays(page)

            # mes actual por label visible (más confiable)
            month_label = get_month_label(page) or "unknown"
            cur_month = label_to_ym(month_label) or "unknown"

            # stats links visibles
            links = get_stats_links(page)
            print(f"\n=== Mes detectado: {cur_month} ({month_label}) ===")
            print(f"Links stats visibles: {len(links)}")

            if cur_month != "unknown" and cur_month not in processed_months:
                processed_months.add(cur_month)

            # carpeta por mes
            month_dir = os.path.join(args.out, cur_month)
            safe_mkdir(month_dir)

            # descargar cada partido (una pestaña por partido)
            for u in links:
                mid = match_id_from_url(u)
                if not mid:
                    continue
                if mid in seen_match_ids:
                    continue
                seen_match_ids.add(mid)

                title = mp.title()
                fname = sanitize_filename(title)

                # Asegura extensión .htm (muchas herramientas la esperan así)
                out_file = os.path.join(month_dir, f"{fname}.htm")
                if os.path.exists(out_file):
                    continue

                mp = context.new_page()
                try:
                    print(f"  -> {mid} {u}")
                    mp.goto(u, wait_until="domcontentloaded", timeout=60000)

                    for _ in range(2):
                        click_privacy_accept(mp)
                        nuke_overlays(mp)
                        mp.wait_for_timeout(250)

                    mp.wait_for_load_state("networkidle", timeout=60000)

                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(mp.content())

                    time.sleep(args.delay)
                except PWTimeoutError:
                    print(f"     !! Timeout {mid}")
                    try:
                        mp.screenshot(path=os.path.join(month_dir, f"{mid}_timeout.png"), full_page=True)
                    except Exception:
                        pass
                finally:
                    mp.close()

            # condición de parada
            if cur_month != "unknown" and cur_month <= args.until:
                print(f"\nLlegué al mes límite {args.until}. Fin.")
                break

            # intentar ir al mes anterior
            prev_btn = page.locator("#dayChangeBtn-prev")
            if prev_btn.count() == 0:
                print("No encontré botón prev. Fin.")
                break

            old_label = get_month_label(page) or ""
            old_month = label_to_ym(old_label) or "unknown"

            changed = False
            for attempt in range(1, args.max_prev_retries + 1):
                click_privacy_accept(page)
                nuke_overlays(page)

                # scroll + click forzado; si falla, click por JS
                try:
                    prev_btn.scroll_into_view_if_needed(timeout=5000)
                except Exception:
                    pass

                try:
                    prev_btn.click(timeout=5000, force=True)
                except Exception:
                    try:
                        page.evaluate("() => document.querySelector('#dayChangeBtn-prev')?.click()")
                    except Exception:
                        pass

                # espera a que cambie el label del mes
                try:
                    wait_month_label_change(page, old_label, timeout_ms=20000)
                    changed = True
                    break
                except Exception:
                    page.wait_for_timeout(800)

            if not changed:
                debug_dump(page, dbg_dir, f"prev_no_change_{old_month}")
                raise RuntimeError(
                    "Hice click prev pero el mes NO cambió. "
                    "Casi siempre es por el modal de privacidad/ads. Revisa _debug/prev_no_change_*.png"
                )

            # pequeño respiro para que pinte fixtures
            page.wait_for_timeout(800)

        browser.close()

if __name__ == "__main__":
    main()