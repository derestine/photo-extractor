#!/usr/bin/env python3
"""Tkinter GUI for the photo extractor.

Pick an input folder (of phone scans), pick an output folder, toggle the
pipeline options, hit Process, and browse the extracted photos as a gallery.
Double-click a thumbnail to open it full-size in your system viewer.

Reuses the pipeline from photo_extractor.py — no duplicated logic.
"""

from __future__ import annotations

import platform
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import photo_extractor as pe

HERE = Path(__file__).resolve().parent
THUMB_PX = 220       # longest edge of a gallery thumbnail
GUTTER = 12
COLUMNS = 4


class ExtractorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Photo Extractor")
        self.geometry("1180x800")
        self.minsize(820, 560)

        self.in_path = tk.StringVar(value=str(HERE / "raw"))
        self.out_path = tk.StringVar(value=str(HERE / "out"))
        self.quality = tk.IntVar(value=95)
        self.opts = {
            "enhance": tk.BooleanVar(value=True),
            "crop_white": tk.BooleanVar(value=True),
            "orient": tk.BooleanVar(value=True),
            "deskew": tk.BooleanVar(value=True),
            "remove_dates": tk.BooleanVar(value=False),
        }
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._processing = False

        self._build_controls()
        self._build_gallery()
        self._build_status()
        self.after(120, self._poll)

    # --- UI construction ----------------------------------------------------
    def _build_controls(self) -> None:
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="Input:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(bar, textvariable=self.in_path, width=46).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(bar, text="Browse…", width=10, command=self._browse_in).grid(row=0, column=2, padx=2)

        ttk.Label(bar, text="Output:").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=(6, 0))
        ttk.Entry(bar, textvariable=self.out_path, width=46).grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(bar, text="Browse…", width=10, command=self._browse_out).grid(row=1, column=2, padx=2, pady=(6, 0))

        opts = ttk.Frame(bar)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        labels = {
            "enhance": "Tone normalize",
            "crop_white": "Trim matte",
            "orient": "Auto-orient (faces)",
            "deskew": "Level tilt",
            "remove_dates": "Remove dates (exp.)",
        }
        for i, (key, label) in enumerate(labels.items()):
            ttk.Checkbutton(opts, text=label, variable=self.opts[key]).grid(row=0, column=i, padx=4, sticky="w")

        actions = ttk.Frame(bar)
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(actions, text="JPEG quality:").pack(side="left", padx=(0, 4))
        ttk.Spinbox(actions, from_=50, to=100, width=5, textvariable=self.quality).pack(side="left")
        self.process_btn = ttk.Button(actions, text="Process", command=self._start)
        self.process_btn.pack(side="left", padx=16)
        ttk.Button(actions, text="Refresh gallery", command=self._reload_gallery).pack(side="left", padx=4)
        ttk.Button(actions, text="Open output folder", command=self._open_out).pack(side="left", padx=4)

        bar.columnconfigure(1, weight=1)

    def _build_gallery(self) -> None:
        frame = ttk.LabelFrame(self, text="Results", padding=4)
        frame.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        self.canvas = tk.Canvas(frame, highlightthickness=0, bg="#1c1c1c")
        vbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.grid_frame = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Mouse-wheel scrolling (mac/Win/Linux).
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        self._placeholder = ttk.Label(self.grid_frame, text="No results yet — choose a folder and press Process.",
                                      foreground="#888", padding=20)
        self._placeholder.grid(row=0, column=0, columnspan=COLUMNS)
        self._photos: list[ImageTk.PhotoImage] = []  # keep refs alive

    def _build_status(self) -> None:
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken", padding=(8, 3)).pack(fill="x")

    # --- callbacks ----------------------------------------------------------
    def _browse_in(self) -> None:
        d = filedialog.askdirectory(initialdir=self.in_path.get() or str(HERE), title="Choose input folder")
        if d:
            self.in_path.set(d)

    def _browse_out(self) -> None:
        d = filedialog.askdirectory(initialdir=self.out_path.get() or str(HERE), title="Choose output folder")
        if d:
            self.out_path.set(d)

    def _open_out(self) -> None:
        out = Path(self.out_path.get())
        out.mkdir(parents=True, exist_ok=True)
        _reveal(out)

    def _on_canvas_resize(self, _event) -> None:
        self.canvas.itemconfigure(self.inner_id, width=self.canvas.winfo_width())

    def _on_wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    # --- processing ---------------------------------------------------------
    def _start(self) -> None:
        if self._processing:
            return
        in_dir = Path(self.in_path.get())
        out_dir = Path(self.out_path.get())
        if not in_dir.exists():
            messagebox.showerror("Missing input", f"Input folder not found:\n{in_dir}")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        files = pe.gather_inputs([in_dir])
        if not files:
            messagebox.showinfo("Nothing to do", f"No supported images in {in_dir}")
            return

        flags = (
            self.opts["enhance"].get(),
            self.opts["crop_white"].get(),
            self.opts["orient"].get(),
            self.opts["deskew"].get(),
            self.opts["remove_dates"].get(),
        )
        quality = int(self.quality.get())

        self._processing = True
        self.process_btn.state(["disabled"])
        self.status.set(f"Processing {len(files)} scan(s)…")
        threading.Thread(target=self._worker, args=(files, out_dir, flags, quality), daemon=True).start()

    def _worker(self, files, out_dir, flags, quality) -> None:
        total = len(files)
        photos = 0
        for i, f in enumerate(files, 1):
            try:
                photos += pe.process_file(f, out_dir, *flags, quality)
            except Exception as exc:  # never let one bad file kill the batch
                self.queue.put(("error", f"{f.name}: {exc}"))
            self.queue.put(("progress", f"({i}/{total})  {f.name}"))
        self.queue.put(("done", (photos, total, out_dir)))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self.status.set(f"Processing… {payload}")
                elif kind == "error":
                    self.status.set(f"Error: {payload}")
                elif kind == "done":
                    photos, total, out_dir = payload
                    self._processing = False
                    self.process_btn.state(["!disabled"])
                    self.status.set(f"Done: {photos} photo(s) from {total} scan(s) → {out_dir}")
                    self._load_gallery(Path(out_dir))
        except queue.Empty:
            pass
        self.after(120, self._poll)

    # --- gallery ------------------------------------------------------------
    def _reload_gallery(self) -> None:
        self._load_gallery(Path(self.out_path.get()))

    def _load_gallery(self, folder: Path) -> None:
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._photos.clear()

        files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".jpg")
        if not files:
            ttk.Label(self.grid_frame, text=f"No JPGs in {folder}", foreground="#888", padding=20).grid(
                row=0, column=0, columnspan=COLUMNS
            )
            return

        for i, p in enumerate(files):
            thumb, dims = _thumbnail(p)
            if thumb is None:
                continue
            self._photos.append(thumb)
            cell = ttk.Frame(self.grid_frame, padding=GUTTER)
            cell.grid(row=i // COLUMNS, column=i % COLUMNS, sticky="n", padx=2, pady=2)
            lbl = ttk.Label(cell, image=thumb, cursor="hand2")
            lbl.image = thumb
            lbl.pack()
            lbl.bind("<Double-Button-1>", lambda e, path=p: _reveal(path))
            info = f"{p.name}\n{dims}"
            ttk.Label(cell, text=info, foreground="#444", justify="center", wraplength=THUMB_PX + 20).pack()


def _thumbnail(path: Path) -> tuple[ImageTk.PhotoImage | None, str]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            im.thumbnail((THUMB_PX, THUMB_PX))
            return ImageTk.PhotoImage(im), f"{w}×{h}"
    except Exception:
        return None, "?"


def _reveal(path: Path) -> None:
    """Open a file or folder in the platform's default viewer."""
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif sysname == "Windows":
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def main() -> int:
    app = ExtractorGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
