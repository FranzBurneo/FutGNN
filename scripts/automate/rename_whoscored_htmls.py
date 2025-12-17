from __future__ import annotations

from pathlib import Path
import argparse
import re
import unicodedata


TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def sanitize_filename(name: str, max_len: int = 180) -> str:
    """Convierte un <title> en nombre de archivo válido para Windows."""
    name = (name or "").strip()

    # Quita tildes/acentos (más seguro para compatibilidad)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Reemplazos típicos problemáticos
    name = name.replace("/", "_").replace("\\", "_")

    # Caracteres inválidos en Windows: <>:"/\|?* y saltos de línea
    name = re.sub(r'[<>:"|?*\r\n\t]', " ", name)

    # Espacios múltiples -> uno
    name = re.sub(r"\s+", " ", name).strip()

    # Limita longitud
    if len(name) > max_len:
        name = name[:max_len].rstrip()

    return name


def unique_path(p: Path) -> Path:
    """Si ya existe el nombre, agrega (2), (3)..."""
    if not p.exists():
        return p

    base = p.with_suffix("")  # sin extensión
    ext = p.suffix
    k = 2
    while True:
        cand = Path(f"{base} ({k}){ext}")
        if not cand.exists():
            return cand
        k += 1


def extract_title(html_text: str) -> str | None:
    m = TITLE_RE.search(html_text)
    if not m:
        return None
    return m.group(1).strip()


def iter_html_files(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = list(folder.rglob("*.htm")) + list(folder.rglob("*.html"))
    else:
        files = list(folder.glob("*.htm")) + list(folder.glob("*.html"))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="Renombra HTML de WhoScored usando el <title>.")
    ap.add_argument("folder", help="Carpeta donde están los htmls (ej: .\\htmls)")
    ap.add_argument("--no-recursive", action="store_true", help="No buscar en subcarpetas")
    ap.add_argument("--dry-run", action="store_true", help="Solo muestra qué haría, no renombra")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"ERROR: No existe la carpeta: {folder}")

    recursive = not args.no_recursive
    files = iter_html_files(folder, recursive=recursive)

    print(f"Carpeta: {folder}")
    print(f"Modo: {'recursivo' if recursive else 'no recursivo'}")
    print(f"Archivos encontrados: {len(files)}")

    if not files:
        print("No hay archivos .htm/.html para procesar.")
        print("Tip: si están dentro de subcarpetas, NO uses --no-recursive.")
        return

    renamed = 0
    skipped_no_title = 0
    skipped_same = 0

    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] No pude leer: {f} -> {e}")
            continue

        title = extract_title(txt)
        if not title:
            skipped_no_title += 1
            # Para depurar, descomenta:
            # print(f"[SKIP] Sin <title>: {f}")
            continue

        new_base = sanitize_filename(title)
        if not new_base:
            skipped_no_title += 1
            continue

        # Normaliza a .htm (más común en dumps)
        target = f.with_name(new_base + ".htm")
        target = unique_path(target)

        if target.name == f.name:
            skipped_same += 1
            continue

        if args.dry_run:
            print(f"[DRY] {f.name} -> {target.name}")
            renamed += 1
            continue

        try:
            f.rename(target)
            print(f"[OK]  {f.name} -> {target.name}")
            renamed += 1
        except Exception as e:
            print(f"[ERR] No pude renombrar {f} -> {target}: {e}")

    print("\nResumen:")
    print(f"  Renombrados: {renamed}")
    print(f"  Sin <title> (omitidos): {skipped_no_title}")
    print(f"  Ya tenían ese nombre (omitidos): {skipped_same}")


if __name__ == "__main__":
    main()