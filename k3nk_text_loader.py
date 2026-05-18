import os
import re


class K3NKTextLoaderFromDir:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Ruta al directorio con archivos .txt"}),
                "sort_method": (["None", "Alphabetical (ASC)", "Alphabetical (DESC)",
                                 "Numerical (ASC)", "Numerical (DESC)",
                                 "Datetime (ASC)", "Datetime (DESC)"],),
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
                                        "tooltip": "Índice del archivo a cargar"}),
                "load_always": ("BOOLEAN", {"default": False,
                                            "label_on": "Enabled", "label_off": "Disabled"}),
                "encoding": (["utf-8", "utf-8-sig", "latin-1", "cp1252"],),
            }
        }

    RETURN_TYPES  = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES  = ("text", "index", "total_files", "filename")
    FUNCTION      = "load_text"
    CATEGORY      = "K3NK/loaders"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("load_always", False):
            return float("NaN")
        return hash(frozenset({k: str(v) for k, v in kwargs.items()}))

    def _sort(self, items, base_path, method):
        def num(s):
            m = re.search(r"\d+", s)
            return int(m.group()) if m else float("inf")
        fp = lambda x: os.path.join(base_path, x)
        if method == "Alphabetical (ASC)":  return sorted(items)
        if method == "Alphabetical (DESC)": return sorted(items, reverse=True)
        if method == "Numerical (ASC)":     return sorted(items, key=lambda x: num(os.path.splitext(x)[0]))
        if method == "Numerical (DESC)":    return sorted(items, key=lambda x: num(os.path.splitext(x)[0]), reverse=True)
        if method == "Datetime (ASC)":      return sorted(items, key=lambda x: os.path.getmtime(fp(x)))
        if method == "Datetime (DESC)":     return sorted(items, key=lambda x: os.path.getmtime(fp(x)), reverse=True)
        return items

    def load_text(self, directory, sort_method="None",
                  start_index=0, load_always=False, encoding="utf-8"):

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directorio no encontrado: '{directory}'")

        txt_files = [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(".txt")
        ]

        if not txt_files:
            raise FileNotFoundError(f"No hay archivos .txt en '{directory}'")

        txt_files = self._sort(txt_files, directory, sort_method)
        total = len(txt_files)

        if start_index >= total:
            raise ValueError(f"start_index={start_index} fuera de rango (hay {total} archivos)")

        chosen    = txt_files[start_index]
        full_path = os.path.join(directory, chosen)

        with open(full_path, "r", encoding=encoding) as f:
            text = f.read()

        print(f"📄 [{start_index + 1}/{total}] {chosen} ({len(text)} chars)")
        return (text, start_index, total, chosen)


NODE_CLASS_MAPPINGS        = {"K3NKTextLoaderFromDir": K3NKTextLoaderFromDir}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKTextLoaderFromDir": "Load Text From Dir (K3NK)"}
