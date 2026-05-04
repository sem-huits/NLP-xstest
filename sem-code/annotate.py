"""
annotate.py  —  XSTest manual annotation tool
Usage: python annotate.py --annotator 1   (or 2)
Output: annotation_1.csv  /  annotation_2.csv
"""

import argparse
import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================
MODELS      = ["neutral", "child", "adult", "elderly"]
SAMPLE_SIZE = 250         # safe prompts per condition (XSTest ratio)
RANDOM_SEED = 42
DATA_DIR    = "."         # folder with *_results.csv files

LABELS = [
    ("1 — Full compliance",    "1_full_compliance"),
    ("2 — Partial compliance", "2_partial_compliance"),
    ("3 — Full refusal",       "3_full_refusal"),
]

# ============================================================================
# BUILD ANNOTATION SAMPLE (safe prompts only)
# ============================================================================
def build_sample():
    frames = []
    for model in MODELS:
        path = os.path.join(DATA_DIR, f"{model}_results.csv")
        df   = pd.read_csv(path)
        safe = df[~df["type"].str.contains("contrast")].copy()

        # stratified sample across prompt types
        n_types = safe["type"].nunique()
        sample  = (
        safe.groupby("type")
            .apply(lambda x: x.sample(
                min(len(x), max(1, SAMPLE_SIZE // n_types)),
                random_state=RANDOM_SEED))
        )

        # 'type' is nu een index-level; zet hem terug als kolom
        sample = sample.reset_index(level=0).reset_index(drop=True)

        # top up to exactly SAMPLE_SIZE if rounding left us short
        if len(sample) < SAMPLE_SIZE:
            remaining = safe.drop(sample.index)
            extra     = remaining.sample(
                min(SAMPLE_SIZE - len(sample), len(remaining)),
                random_state=RANDOM_SEED)
            sample = pd.concat([sample, extra])

        sample = sample.head(SAMPLE_SIZE).copy()
        frames.append(sample)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["id", "type", "age_condition", "prompt", "completion"]]
    combined["annotation"] = ""
    return combined.reset_index(drop=True)


# ============================================================================
# GUI
# ============================================================================
class AnnotationApp:

    def __init__(self, root, df, output_path):
        self.root        = root
        self.df          = df
        self.output_path = output_path
        self.current     = 0

        # jump to first unannotated row
        unannotated  = self.df[self.df["annotation"] == ""].index
        self.current = int(unannotated[0]) if len(unannotated) else len(self.df)

        self.root.title("XSTest Annotation Tool")
        self.root.geometry("900x680")
        self.root.configure(bg="#f5f5f5")
        self._build_ui()
        self._load_item()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        top = tk.Frame(self.root, bg="#f5f5f5")
        top.pack(fill="x", padx=20, pady=(15, 0))

        self.lbl_progress = tk.Label(top, text="", font=("Helvetica", 11),
                                     bg="#f5f5f5", fg="#555")
        self.lbl_progress.pack(side="left")

        self.lbl_type = tk.Label(top, text="", font=("Helvetica", 11, "italic"),
                                 bg="#f5f5f5", fg="#888")
        self.lbl_type.pack(side="right")

        # prompt box
        tk.Label(self.root, text="PROMPT", font=("Helvetica", 10, "bold"),
                 bg="#f5f5f5", anchor="w").pack(fill="x", padx=20, pady=(12, 2))
        self.txt_prompt = tk.Text(self.root, height=4, wrap="word",
                                  font=("Helvetica", 12), state="disabled",
                                  bg="#eef2ff", relief="flat", padx=8, pady=6)
        self.txt_prompt.pack(fill="x", padx=20)

        # completion box
        tk.Label(self.root, text="COMPLETION", font=("Helvetica", 10, "bold"),
                 bg="#f5f5f5", anchor="w").pack(fill="x", padx=20, pady=(12, 2))
        frame_comp = tk.Frame(self.root, bg="#f5f5f5")
        frame_comp.pack(fill="both", expand=True, padx=20)

        scrollbar = tk.Scrollbar(frame_comp)
        scrollbar.pack(side="right", fill="y")

        self.txt_completion = tk.Text(frame_comp, wrap="word",
                                      font=("Helvetica", 12), state="disabled",
                                      bg="#ffffff", relief="flat", padx=8, pady=6,
                                      yscrollcommand=scrollbar.set)
        self.txt_completion.pack(fill="both", expand=True)
        scrollbar.config(command=self.txt_completion.yview)

        # label buttons
        btn_frame = tk.Frame(self.root, bg="#f5f5f5")
        btn_frame.pack(pady=18)

        colors = ["#4caf50", "#ff9800", "#f44336"]
        self.btns = []
        for i, (label_text, label_val) in enumerate(LABELS):
            btn = tk.Button(
                btn_frame,
                text=f"[{i+1}]  {label_text}",
                font=("Helvetica", 13, "bold"),
                bg=colors[i], fg="white",
                activebackground=colors[i],
                relief="flat", padx=20, pady=10,
                command=lambda v=label_val: self._annotate(v)
            )
            btn.pack(side="left", padx=10)
            self.btns.append(btn)

        # keyboard shortcuts
        self.root.bind("1", lambda e: self._annotate("1_full_compliance"))
        self.root.bind("2", lambda e: self._annotate("2_partial_compliance"))
        self.root.bind("3", lambda e: self._annotate("3_full_refusal"))

        # back button
        self.btn_back = tk.Button(self.root, text="← Back",
                                  font=("Helvetica", 10), bg="#ddd",
                                  relief="flat", padx=10, pady=5,
                                  command=self._go_back)
        self.btn_back.pack(pady=(0, 10))

    # --------------------------------------------------------------- logic --
    def _load_item(self):
        if self.current >= len(self.df):
            messagebox.showinfo("Done", "All items annotated! File saved.")
            self.root.quit()
            return

        row   = self.df.iloc[self.current]
        total = len(self.df)
        done  = (self.df["annotation"] != "").sum()

        self.lbl_progress.config(
            text=f"Item {self.current + 1} / {total}  |  Done: {done}")
        self.lbl_type.config(
            text=f"{row['age_condition']}  ·  {row['type']}")

        self._set_text(self.txt_prompt,     row["prompt"])
        self._set_text(self.txt_completion, row["completion"])

        # highlight already-annotated button
        existing = row["annotation"]
        for btn, (_, val) in zip(self.btns, LABELS):
            btn.config(relief="sunken" if val == existing else "flat")

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", str(text))
        widget.config(state="disabled")
        widget.yview_moveto(0)

    def _annotate(self, label_value):
        self.df.at[self.current, "annotation"] = label_value
        self._save()
        self.current += 1
        self._load_item()

    def _go_back(self):
        if self.current > 0:
            self.current -= 1
            self._load_item()

    def _save(self):
        self.df.to_csv(self.output_path, index=False)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", type=int, required=True, choices=[1, 2],
                        help="Annotator number (1 or 2)")
    args = parser.parse_args()

    output_path = f"annotation_{args.annotator}.csv"

    # build or load sample
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        df["annotation"] = df["annotation"].fillna("")
        print(f"Resuming from {output_path} "
              f"({(df['annotation'] != '').sum()}/{len(df)} done)")
    else:
        print("Building annotation sample...")
        df = build_sample()
        df.to_csv(output_path, index=False)
        print(f"Sample saved to {output_path} ({len(df)} items)")

    root = tk.Tk()
    AnnotationApp(root, df, output_path)
    root.mainloop()


if __name__ == "__main__":
    main()
