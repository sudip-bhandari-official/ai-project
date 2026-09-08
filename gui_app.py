"""
Graphical User Interface for Smart Crop Recommendation (gui_app.py)
-------------------------------------------------------------------
Responsibility:
- Provides a clean, modern, native desktop interface using Python's Tkinter.
- Allows farmers and agronomists to input 7 soil and climate variables.
- Features a responsive, native horizontal bar graph ranking matchable crops by percentage.
- Displays multi-crop recommendations with suitability tiers and agronomic comparative insights.
- Offers 1-click individual crop presets for rapid testing of diverse conditions.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from data_prep import load_and_preprocess_data, FEATURE_METADATA, FEATURE_NAMES
from model_trainer import train_gaussian_nb, predict_crop, format_crop_name

# Practical Individual Crop Presets (No '&' or compound names)
PRESETS = {
    "Rice": [80.0, 48.0, 40.0, 24.0, 82.0, 6.4, 236.0],
    "Litchi": [62.0, 32.0, 50.0, 26.8, 81.5, 6.3, 188.0],
    "Watermelon": [99.0, 17.0, 50.0, 25.6, 85.2, 6.5, 50.8],
    "Mango": [20.0, 27.0, 30.0, 31.2, 50.2, 5.8, 94.7],
    "Wheat": [85.0, 40.0, 35.0, 18.5, 55.0, 6.8, 75.0],
    "Cotton": [118.0, 46.0, 20.0, 24.0, 79.8, 6.9, 80.4],
    "Apple": [21.0, 134.0, 199.0, 22.6, 92.3, 5.9, 112.7],
    "Coffee": [101.0, 29.0, 30.0, 25.5, 58.9, 6.8, 158.1],
}

CROP_ICONS = {
    "rice": "🌾", "wheat": "🌾", "maize": "🌽", "barley": "🌾", "oats": "🌾",
    "litchi": "🍒", "mango": "🥭", "watermelon": "🍉", "muskmelon": "🍈",
    "apple": "🍎", "banana": "🍌", "grapes": "🍇", "orange": "🍊",
    "papaya": "🍈", "pomegranate": "🍎", "guava": "🍐", "pineapple": "🍍",
    "strawberry": "🍓", "peach": "🍑", "plum": "🫐", "cherry": "🍒",
    "potato": "🥔", "tomato": "🍅", "onion": "🧅", "garlic": "🧄",
    "ginger": "🫚", "carrot": "🥕", "cauliflower": "🥦", "cabbage": "🥬",
    "spinach": "🥬", "brinjal": "🍆", "okra": "🌱", "cucumber": "🥒",
    "cotton": "🌿", "jute": "🌾", "coffee": "☕", "tea": "🍵",
    "sugarcane": "🎋", "coconut": "🥥", "cocoa": "🍫", "cashew": "🥜",
    "chickpea": "🫘", "lentil": "🥣", "soybean": "🫘", "groundnut": "🥜",
    "sunflower": "🌻", "cardamom": "🌿", "turmeric": "🫚", "clove": "🌿",
    "jackfruit": "🍈", "dragonfruit": "🐉", "avocado": "🥑", "fig": "🫐",
}


class CropChartCanvas(tk.Canvas):
    """
    Custom Canvas Widget that renders a modern, responsive horizontal bar chart
    showing all matchable crops ranked by suitability percentage.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#ffffff", highlightthickness=0, **kwargs)
        self.candidates = []
        self.bind("<Configure>", lambda e: self.redraw())

    def set_data(self, candidates):
        """
        Updates chart data and triggers redraw.
        candidates: list of tuples [(crop, percentage, badge), ...]
        """
        self.candidates = candidates or []
        self.redraw()

    def redraw(self):
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 100 or height < 80:
            return

        # Background grid and padding
        pad_left = 145
        pad_right = 100
        pad_top = 20
        pad_bottom = 26

        chart_w = max(50, width - pad_left - pad_right)
        chart_h = max(40, height - pad_top - pad_bottom)

        # Draw vertical grid lines for 0%, 25%, 50%, 75%, 100%
        grid_steps = [0, 25, 50, 75, 100]
        for pct in grid_steps:
            x = pad_left + (pct / 100.0) * chart_w
            self.create_line(x, pad_top - 4, x, pad_top + chart_h, fill="#f1f5f9", dash=(2, 2))
            self.create_text(
                x, pad_top + chart_h + 12,
                text=f"{pct}%", font=("Segoe UI", 8), fill="#94a3b8"
            )

        if not self.candidates:
            self.create_text(
                width / 2, height / 2,
                text="Click 'Recommend Optimal Crops' to view comparison graph",
                font=("Segoe UI", 10, "italic"), fill="#94a3b8"
            )
            return

        num_bars = len(self.candidates)
        bar_gap = 6
        bar_height = max(18, min(26, int((chart_h - (num_bars - 1) * bar_gap) / num_bars)))

        for i, (crop, pct, badge) in enumerate(self.candidates):
            y0 = pad_top + i * (bar_height + bar_gap)
            y1 = y0 + bar_height

            base_crop = crop.split("_")[0].lower()
            icon = CROP_ICONS.get(base_crop, "🌱")
            disp_name = format_crop_name(crop)
            crop_display = f"#{i + 1} {icon} {disp_name}"

            # Crop Name Label on left
            self.create_text(
                pad_left - 8, (y0 + y1) / 2,
                text=crop_display,
                font=("Segoe UI", 9, "bold" if i == 0 else "normal"),
                anchor="e",
                fill="#0f172a" if i == 0 else "#334155"
            )

            # Background bar track
            self.create_rectangle(
                pad_left, y0, pad_left + chart_w, y1,
                fill="#f8fafc", outline="#e2e8f0"
            )

            # Filled bar width
            fill_w = (max(0.0, min(100.0, pct)) / 100.0) * chart_w

            # Color coding based on match score
            if pct >= 85.0:
                bar_color = "#059669"  # Emerald
            elif pct >= 70.0:
                bar_color = "#10b981"  # Mint green
            elif pct >= 55.0:
                bar_color = "#f59e0b"  # Amber
            elif pct >= 40.0:
                bar_color = "#3b82f6"  # Blue
            else:
                bar_color = "#cbd5e1"  # Slate

            if fill_w > 0:
                self.create_rectangle(
                    pad_left, y0, pad_left + fill_w, y1,
                    fill=bar_color, outline=""
                )

            # Percentage label on right
            pct_text = f"{pct:.1f}%"
            self.create_text(
                pad_left + chart_w + 8, (y0 + y1) / 2,
                text=pct_text,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                fill=bar_color if i == 0 else "#1e293b"
            )

            # Small status badge tag
            badge_short = badge.replace(" Match", "")
            self.create_text(
                pad_left + chart_w + 54, (y0 + y1) / 2,
                text=f"[{badge_short}]",
                font=("Segoe UI", 8),
                anchor="w",
                fill="#64748b"
            )


class CropRecommendationApp:
    def __init__(self, root, model=None):
        self.root = root
        self.root.title("Smart Crop Recommendation System (1,000 Crops Catalog)")
        self.root.geometry("1060x750")
        self.root.minsize(980, 700)

        # Style & theme setup
        self._setup_styles()

        # Model initialization
        if model is None:
            X_train, X_test, y_train, y_test = load_and_preprocess_data()
            self.model, self.accuracy, _ = train_gaussian_nb(X_train, X_test, y_train, y_test, verbose=False)
        else:
            self.model = model
            self.accuracy = getattr(model, "test_accuracy_", 0.99)

        # Tkinter input variables
        self.entries = {}

        # Build UI sections
        self._build_header()
        self._build_main_content()
        self._build_footer()

        # Apply default preset (Litchi)
        self._apply_preset("Litchi")

    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Clean color palette
        self.bg_main = "#f8fafc"
        self.card_bg = "#ffffff"
        self.border_color = "#e2e8f0"
        self.primary_green = "#059669"
        self.primary_hover = "#047857"
        self.text_dark = "#0f172a"
        self.text_muted = "#64748b"

        self.root.configure(bg=self.bg_main)

        # Global styles
        self.style.configure(".", background=self.bg_main, foreground=self.text_dark, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#065f46", background=self.bg_main)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground=self.text_muted, background=self.bg_main)
        self.style.configure("CardTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground="#0f172a", background=self.card_bg)

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=self.bg_main, padx=20, pady=12)
        header_frame.pack(fill="x")

        title_row = tk.Frame(header_frame, bg=self.bg_main)
        title_row.pack(fill="x")

        title_lbl = tk.Label(
            title_row,
            text="🌱 Smart Crop Recommendation System",
            font=("Segoe UI", 16, "bold"),
            fg="#065f46",
            bg=self.bg_main
        )
        title_lbl.pack(side="left")

        badge_lbl = tk.Label(
            title_row,
            text="1,000 Crops Catalog • High-Speed AI Engine",
            font=("Segoe UI", 8, "bold"),
            bg="#d1fae5",
            fg="#065f46",
            padx=8,
            pady=2,
            relief="flat"
        )
        badge_lbl.pack(side="left", padx=12)

        sub_lbl = tk.Label(
            header_frame,
            text="Analyze soil nutrients and environmental parameters to find optimal and matchable crops with confidence percentages.",
            font=("Segoe UI", 9),
            fg=self.text_muted,
            bg=self.bg_main
        )
        sub_lbl.pack(anchor="w", pady=(3, 0))

        divider = tk.Frame(header_frame, height=1, bg="#e2e8f0")
        divider.pack(fill="x", pady=(8, 0))

    def _build_main_content(self):
        content_frame = tk.Frame(self.root, bg=self.bg_main, padx=16, pady=6)
        content_frame.pack(fill="both", expand=True)

        content_frame.columnconfigure(0, weight=4)  # Left inputs: 40%
        content_frame.columnconfigure(1, weight=6)  # Right output: 60%
        content_frame.rowconfigure(0, weight=1)

        # ----------------- LEFT CARD: Inputs -----------------
        left_card = tk.Frame(content_frame, bg=self.card_bg, padx=16, pady=14, relief="solid", bd=1)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(
            left_card,
            text="Soil & Climate Parameters",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_dark,
            bg=self.card_bg
        ).pack(anchor="w", pady=(0, 8))

        # Quick Presets Bar (Individual crops only, no '&')
        preset_box = tk.LabelFrame(
            left_card,
            text=" Quick Crop Presets ",
            font=("Segoe UI", 8, "bold"),
            fg=self.text_muted,
            bg=self.card_bg,
            padx=8,
            pady=6
        )
        preset_box.pack(fill="x", pady=(0, 10))

        preset_row1 = tk.Frame(preset_box, bg=self.card_bg)
        preset_row1.pack(fill="x", pady=2)
        preset_row2 = tk.Frame(preset_box, bg=self.card_bg)
        preset_row2.pack(fill="x", pady=2)

        preset_names = list(PRESETS.keys())
        for idx, name in enumerate(preset_names):
            parent_row = preset_row1 if idx < 4 else preset_row2
            btn = tk.Button(
                parent_row,
                text=name,
                font=("Segoe UI", 8, "bold"),
                bg="#f0fdf4",
                fg="#15803d",
                activebackground="#dcfce7",
                relief="groove",
                bd=1,
                padx=6,
                pady=3,
                cursor="hand2",
                command=lambda p=name: self._apply_preset(p)
            )
            btn.pack(side="left", padx=2, expand=True, fill="x")

        # Form Inputs Grid
        form_frame = tk.Frame(left_card, bg=self.card_bg)
        form_frame.pack(fill="both", expand=True, pady=4)

        for feat in FEATURE_NAMES:
            meta = FEATURE_METADATA[feat]

            row = tk.Frame(form_frame, bg=self.card_bg)
            row.pack(fill="x", pady=4)

            lbl = tk.Label(
                row,
                text=f"{meta['label']}:",
                font=("Segoe UI", 9, "bold"),
                bg=self.card_bg,
                fg="#1e293b",
                anchor="w",
                width=17
            )
            lbl.pack(side="left")

            var = tk.StringVar()
            entry = tk.Entry(
                row,
                textvariable=var,
                font=("Segoe UI", 9),
                width=9,
                relief="solid",
                bd=1,
                bg="#f8fafc",
                highlightthickness=1,
                highlightcolor=self.primary_green
            )
            entry.pack(side="left", padx=(4, 8))
            self.entries[feat] = var

            hint_text = f"{meta['min']} - {meta['max']} {meta['unit']}"
            hint_lbl = tk.Label(
                row,
                text=hint_text,
                font=("Segoe UI", 8),
                fg=self.text_muted,
                bg=self.card_bg
            )
            hint_lbl.pack(side="left")

        # Action Buttons
        btn_frame = tk.Frame(left_card, bg=self.card_bg)
        btn_frame.pack(fill="x", pady=(12, 4))

        predict_btn = tk.Button(
            btn_frame,
            text="🌱 Recommend Optimal Crops",
            font=("Segoe UI", 10, "bold"),
            bg=self.primary_green,
            fg="#ffffff",
            activebackground=self.primary_hover,
            activeforeground="#ffffff",
            relief="flat",
            pady=8,
            cursor="hand2",
            command=self.on_predict
        )
        predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        reset_btn = tk.Button(
            btn_frame,
            text="↺ Clear",
            font=("Segoe UI", 9),
            bg="#f1f5f9",
            fg="#475569",
            activebackground="#e2e8f0",
            relief="flat",
            pady=8,
            padx=12,
            cursor="hand2",
            command=self._clear_entries
        )
        reset_btn.pack(side="right")

        # ----------------- RIGHT CARD: Results & Graph -----------------
        right_card = tk.Frame(content_frame, bg=self.card_bg, padx=16, pady=14, relief="solid", bd=1)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        tk.Label(
            right_card,
            text="Recommendation & Comparison Analysis",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_dark,
            bg=self.card_bg
        ).pack(anchor="w", pady=(0, 8))

        # Hero Best Match Card
        self.hero_frame = tk.Frame(right_card, bg="#ecfdf5", padx=14, pady=10, relief="solid", bd=1)
        self.hero_frame.pack(fill="x", pady=(0, 10))

        hero_top_row = tk.Frame(self.hero_frame, bg="#ecfdf5")
        hero_top_row.pack(fill="x")

        self.hero_icon_lbl = tk.Label(
            hero_top_row,
            text="🌱",
            font=("Segoe UI", 24),
            bg="#ecfdf5"
        )
        self.hero_icon_lbl.pack(side="left", padx=(0, 8))

        hero_text_box = tk.Frame(hero_top_row, bg="#ecfdf5")
        hero_text_box.pack(side="left", fill="x", expand=True)

        self.hero_crop_lbl = tk.Label(
            hero_text_box,
            text="RECOMMENDED CROP: --",
            font=("Segoe UI", 13, "bold"),
            fg="#065f46",
            bg="#ecfdf5",
            anchor="w"
        )
        self.hero_crop_lbl.pack(anchor="w")

        self.hero_match_lbl = tk.Label(
            hero_text_box,
            text="Match Score: -- % • Optimal Growing Conditions",
            font=("Segoe UI", 9, "bold"),
            fg="#047857",
            bg="#ecfdf5",
            anchor="w"
        )
        self.hero_match_lbl.pack(anchor="w", pady=(2, 0))

        # Graph Header & Chart Frame
        graph_header = tk.Frame(right_card, bg=self.card_bg)
        graph_header.pack(fill="x", pady=(2, 4))

        tk.Label(
            graph_header,
            text="📊 Related Crops Suitability Graph (Highest Match on Top):",
            font=("Segoe UI", 9, "bold"),
            fg="#1e293b",
            bg=self.card_bg
        ).pack(side="left")

        tk.Label(
            graph_header,
            text="Scale: 0 - 100%",
            font=("Segoe UI", 8),
            fg=self.text_muted,
            bg=self.card_bg
        ).pack(side="right")

        # Custom Responsive Bar Chart Canvas
        self.chart_canvas = CropChartCanvas(right_card, height=210)
        self.chart_canvas.pack(fill="both", expand=True, pady=(0, 8))

        # Agronomic Insight Box
        insight_frame = tk.Frame(right_card, bg="#f8fafc", padx=12, pady=8, relief="solid", bd=1)
        insight_frame.pack(fill="x", side="bottom")

        tk.Label(
            insight_frame,
            text="💡 Environmental Compatibility Insight:",
            font=("Segoe UI", 8, "bold"),
            fg="#0f172a",
            bg="#f8fafc"
        ).pack(anchor="w", pady=(0, 2))

        self.insight_lbl = tk.Label(
            insight_frame,
            text="Enter soil & weather values to calculate multi-crop compatibility.",
            font=("Segoe UI", 8),
            fg="#475569",
            bg="#f8fafc",
            wraplength=520,
            justify="left"
        )
        self.insight_lbl.pack(anchor="w")

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=self.bg_main, padx=20, pady=6)
        footer.pack(fill="x", side="bottom")

        status_text = f"Status: Model Ready • 1,000 Agricultural Crop Varieties Profiled • Inference: <2ms"
        tk.Label(
            footer,
            text=status_text,
            font=("Segoe UI", 8),
            fg=self.text_muted,
            bg=self.bg_main
        ).pack(side="left")

    def _apply_preset(self, preset_name):
        values = PRESETS.get(preset_name, [])
        for feat, val in zip(FEATURE_NAMES, values):
            self.entries[feat].set(str(val))
        self.on_predict()

    def _clear_entries(self):
        for var in self.entries.values():
            var.set("")
        self.hero_icon_lbl.config(text="🌱")
        self.hero_crop_lbl.config(text="RECOMMENDED CROP: --")
        self.hero_match_lbl.config(text="Match Score: -- %")
        self.chart_canvas.set_data([])
        self.insight_lbl.config(text="Enter soil & weather values to calculate multi-crop compatibility.")

    def on_predict(self):
        # Validate inputs
        input_vals = []
        for feat in FEATURE_NAMES:
            raw = self.entries[feat].get().strip()
            if not raw:
                messagebox.showwarning("Input Missing", f"Please enter a value for '{FEATURE_METADATA[feat]['label']}'.")
                return
            try:
                val = float(raw)
                input_vals.append(val)
            except ValueError:
                messagebox.showerror("Invalid Input", f"The value for '{FEATURE_METADATA[feat]['label']}' must be a valid number.")
                return

        try:
            res = predict_crop(self.model, input_vals, top_k=6)
            top_raw_crop = res["recommended_crop"]
            top_crop = format_crop_name(top_raw_crop)
            confidence = res["confidence"]
            base_crop = top_raw_crop.split("_")[0].lower()
            icon = CROP_ICONS.get(base_crop, "🌱")

            # Update Hero Card
            self.hero_icon_lbl.config(text=icon)
            self.hero_crop_lbl.config(text=f"BEST MATCH: {icon} {top_crop.upper()}")
            badge_tier = res["top_candidates"][0][2]
            self.hero_match_lbl.config(text=f"Suitability Score: {confidence:.1f}% • {badge_tier}")

            # Update Chart Canvas with top candidates
            self.chart_canvas.set_data(res["top_candidates"])

            # Update Insight
            self.insight_lbl.config(text=res["insight"])

        except Exception as e:
            messagebox.showerror("Inference Error", f"An error occurred during prediction:\n{str(e)}")


def launch_gui(model=None):
    root = tk.Tk()
    app = CropRecommendationApp(root, model=model)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
