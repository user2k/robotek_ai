"""Widok mapy i kamery robota w Tkinter."""
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from queue import Empty, Full, Queue
from copy import deepcopy
from threading import Event, Thread
from time import monotonic
import random
import json
import csv

from episode import Action, Episode
from maps import DEMO_MAP, generate_maze
from player import ROBOT_RADIUS
from world import Terrain
from curriculum import load_plan, level_options, level_rows
from map_designer import MapDesigner
from sampling import validate_sampling
from sampling_gui import SamplingDialog, load_settings
from camera import camera_ppm, CAMERA_ANGLE, CAMERA_DEPTH
from tensor_camera import TensorCamera, camera_pose
from resources import ResourceMonitor, format_resources
from config.config import MODEL_PATH, MODEL_VERSION

COLORS = {
    Terrain.WALL: ("#475569", "Ściana"),
    Terrain.GROUND: ("#b8a184", "Ziemia · 1×"),
    Terrain.SWAMP: ("#657e36", "Bagno · 0.5×"),
    Terrain.FIRE: ("#ed703c", "Ogień · 2 m"),
    Terrain.WATER: ("#3497cd", "Woda · 0.8× / 3 m"),
    Terrain.PAVEMENT: ("#b8c5d5", "Bruk · 1.2×"),
    Terrain.START: ("#39aa88", "START"),
    Terrain.END: ("#b18ae8", "END · cel"),
}
CELL = 48


class GameApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.ai_running = False
        self.ai_callback = None
        self.agent = None
        self.training_thread = None
        self.work_kind = None
        self.training_stop = Event()
        self.events = Queue()
        self.preview_frames = Queue(maxsize=1)
        self.watch_training = Event()
        self.validation_requests = Queue()
        self.interventions = Queue()
        self.training_bots = []
        self.live_metadata = {}
        self.validation_history = []
        self.selected_bot = tk.StringVar(value='1')
        self.swatter = tk.BooleanVar(value=False)
        self.preview_delay = 0.1
        self.last_preview_time = 0.0
        self.sampling_settings = None
        self.map_seed = random.SystemRandom().randrange(2 ** 31)
        self.current_options = {}
        self.current_level_name = ""
        self.current_rows = generate_maze(seed=self.map_seed)
        root.title("Roboty — trening, podgląd 3D i walidacje")
        root.configure(bg="#101827")
        root.resizable(False, False)
        tk.Label(root, text=f"ŚWIAT 2D branch: {MODEL_VERSION}", font=("Segoe UI", 22, "bold"),
                 bg="#101827", fg="white").pack(pady=(16, 2))
        tk.Label(root, text="W / ↑: przód    S / ↓: tył    A / D: obrót    R: restart",
                 font=("Segoe UI", 11), bg="#101827", fg="#cbd5e1").pack(pady=(0, 12))
        map_controls = tk.Frame(root, bg="#101827")
        map_controls.pack(pady=(0, 8))
        tk.Button(map_controls, text="Nowy labirynt", command=self.new_maze).pack(side="left", padx=4)
        tk.Button(map_controls, text="Mapa demonstracyjna", command=self.show_demo).pack(side="left", padx=4)
        tk.Button(map_controls, text="Projektant poziomów", command=self.open_designer).pack(side="left", padx=4)
        self.map_info = tk.Label(map_controls, bg="#101827", fg="#cbd5e1")
        self.map_info.pack(side="left", padx=8)
        views = tk.Frame(root, bg="#101827")
        views.pack(padx=20)
        self.canvas = tk.Canvas(views, width=len(self.current_rows[0]) * CELL, height=len(self.current_rows) * CELL,
                                bg="#101827", highlightthickness=0)
        self.canvas.pack(side="left")
        self.canvas.bind('<Button-1>', self.click_bot)
        camera_panel = tk.Frame(views, bg="#101827")
        camera_panel.pack(side="left", padx=(16, 0), anchor="n")
        tk.Label(camera_panel, text="OCZY ROBOTA · 3D", bg="#101827", fg="white",
                 font=("Segoe UI", 12, "bold")).pack(pady=(0, 8))
        bot_controls = tk.Frame(camera_panel, bg='#101827')
        bot_controls.pack()
        tk.Label(bot_controls, text='Robot:', bg='#101827', fg='white').pack(side='left')
        self.bot_choice = ttk.Combobox(bot_controls, textvariable=self.selected_bot,
                                       values=('1',), state='readonly', width=5)
        self.bot_choice.pack(side='left', padx=5)
        self.bot_choice.bind('<<ComboboxSelected>>', self.select_bot)
        tk.Checkbutton(bot_controls, text='Łapka −20', variable=self.swatter,
                       command=self.update_swatter, bg='#101827', fg='white',
                       selectcolor='#101827').pack(side='left')
        camera_screen = tk.Frame(camera_panel, width=320, height=240, bg="#131924")
        camera_screen.pack()
        camera_screen.pack_propagate(False)
        self.camera_view = tk.Label(camera_screen, bg="#131924", fg="#aab8cb", bd=0)
        self.camera_view.pack(fill="both", expand=True)
        self.camera_enabled = tk.BooleanVar(value=True)
        self._preview_world = None
        self._preview_camera = None
        tk.Checkbutton(camera_panel, text="Pokaż kamerę", variable=self.camera_enabled,
                       command=self.draw, bg="#101827", fg="white", selectcolor="#101827").pack()
        tk.Label(camera_panel, text=f"Kamera {CAMERA_ANGLE:g}° · zasięg {CAMERA_DEPTH:g} m\nFioletowa szachownica = END",
                 bg="#101827", fg="#aab8cb").pack(pady=8)
        self.resource_label = tk.Label(camera_panel, text="Odczyt CPU / RAM / GPU…",
                                       bg="#101827", fg="#aab8cb", font=("Segoe UI", 10),
                                       width=38, height=7, anchor="nw", justify="left", wraplength=310)
        self.resource_label.pack(pady=(8, 0))
        self.resource_monitor = ResourceMonitor()
        dashboard_panel = tk.Frame(views, bg='#101827')
        dashboard_panel.pack(side='left', padx=(16, 0), anchor='n')
        tk.Label(dashboard_panel, text='Q I WALIDACJE', bg='#101827', fg='white',
                 font=('Segoe UI', 12, 'bold')).pack(pady=(0, 8))
        self.q_label = tk.Label(dashboard_panel, text='Q aktualne: — · brak walidacji',
                                bg='#101827', fg='#ffe080', justify='left', font=('Segoe UI', 13, 'bold'))
        self.q_label.pack(pady=4)
        self.q_average_label = tk.Label(dashboard_panel, text='Q średnie · 0/4 walidacji co 1000',
                                        bg='#101827', fg='#aab8cb', justify='left', font=('Segoe UI', 9))
        self.q_average_label.pack()
        tk.Label(dashboard_panel, text='Q = wygrane / mapy · ● co 1000 / ◆ ręczna',
                 bg='#101827', fg='white', font=('Segoe UI', 9)).pack(pady=(6, 0))
        self.validation_graph = tk.Canvas(dashboard_panel, width=310, height=90,
                                          bg='#172235', highlightthickness=0)
        self.validation_graph.pack()
        self.validation_queue_label = tk.Label(dashboard_panel, text='Kolejka walidacji: 0',
                                               bg='#101827', fg='#aab8cb')
        self.validation_queue_label.pack()
        log_panel = tk.Frame(dashboard_panel, bg='#101827')
        log_panel.pack()
        self.validation_log = ttk.Treeview(log_panel, columns=('group', 'kind', 'wins', 'q', 'score'),
                                           show='headings', height=8)
        for key, label, width in [('group', 'Grupa', 50), ('kind', 'Typ', 55),
                                   ('wins', 'Wygrane', 60), ('q', 'Q', 60), ('score', 'Wynik', 65)]:
            self.validation_log.heading(key, text=label)
            self.validation_log.column(key, width=width, anchor='center', stretch=False)
        self.validation_log.pack(side='left')
        log_scroll = ttk.Scrollbar(log_panel, orient='vertical', command=self.validation_log.yview)
        log_scroll.pack(side='left', fill='y')
        self.validation_log.configure(yscrollcommand=log_scroll.set)
        tk.Label(dashboard_panel, text='Kliknij bota, aby przełączyć kamerę.\nŁapka: kliknięcie kończy przebieg z karą −20.',
                 bg='#101827', fg='#aab8cb', font=('Segoe UI', 9)).pack(pady=8)
        self.load_validation_history()
        legend = tk.Frame(root, bg="#101827")
        legend.pack(pady=12)
        for i, (color, label) in enumerate(COLORS.values()):
            tk.Label(legend, text="  " + label + "  ", bg=color, fg="#101827",
                     font=("Segoe UI", 10, "bold")).grid(row=i // 4, column=i % 4, padx=3, pady=3)
        status_panel = tk.Frame(root, bg="#101827", width=1060, height=124)
        status_panel.pack(padx=20, pady=(0, 8))
        status_panel.pack_propagate(False)
        self.status = tk.Label(status_panel, bg="#101827", fg="white",
                               font=("Segoe UI", 11), wraplength=1040, anchor="center")
        self.status.pack(fill="both", expand=True)
        self.movement_label = tk.Label(root, bg="#101827", fg="#aab8cb", font=("Segoe UI", 10))
        self.movement_label.pack(pady=(0, 12))
        controls = tk.Frame(root, bg="#101827")
        controls.pack()
        tk.Label(controls, text="Grupy / mapy:", bg="#101827", fg="white").pack(side="left")
        self.episode_count = tk.StringVar(value="300")
        tk.Spinbox(controls, from_=1, to=1_000_000, increment=50, width=8,
                   textvariable=self.episode_count).pack(side="left", padx=6)
        self.train_button = tk.Button(controls, text="Trenuj dalej", command=self.start_training)
        self.train_button.pack(side="left", padx=4)
        self.ai_button = tk.Button(controls, text="Uruchom AI", command=self.start_ai)
        self.ai_button.pack(side="left", padx=4)
        self.validation_button = tk.Button(controls, text="Walidacja", command=self.start_validation)
        self.validation_button.pack(side="left", padx=4)
        tk.Button(controls, text="Stop", command=self.stop_work).pack(side="left", padx=4)
        tk.Button(controls, text="Graj sam / restart", command=self.reset).pack(side="left", padx=4)
        batch_controls = tk.Frame(root, bg="#101827")
        batch_controls.pack(pady=(6, 0))
        tk.Label(batch_controls, text="Roboty na mapę (batch):", bg="#101827", fg="white").pack(side="left")
        self.batch_count = tk.StringVar(value="5")
        self.batch_input = tk.Spinbox(batch_controls, from_=1, to=128, width=5, textvariable=self.batch_count)
        self.batch_input.pack(side="left", padx=6)
        tk.Label(batch_controls, text="Urządzenie:", bg="#101827", fg="white").pack(side="left")
        self.device_choice = tk.StringVar(value="auto")
        self.device_input = tk.OptionMenu(batch_controls, self.device_choice, "auto", "cuda", "cpu")
        self.device_input.pack(side="left", padx=6)
        tk.Label(batch_controls, text="1 mapa → N przebiegów → wybór → 1 BPTT", bg="#101827", fg="#aab8cb").pack(side="left")
        tk.Label(batch_controls, text="Start od lvl:", bg="#101827", fg="white").pack(side="left", padx=(8, 0))
        self.start_level_choice = tk.StringVar(value="Kontynuuj")
        self.start_level_input = ttk.Combobox(batch_controls, textvariable=self.start_level_choice,
                                              state="readonly", width=11,
                                              postcommand=self.refresh_start_levels)
        self.start_level_input.pack(side="left", padx=6)
        self.refresh_start_levels()
        watch_controls = tk.Frame(root, bg="#101827")
        watch_controls.pack(pady=(6, 0))
        self.watch_button = tk.Button(watch_controls, text="Trenuj i oglądaj", command=self.start_watching)
        self.watch_button.pack(side="left", padx=4)
        self.sampling_button = tk.Button(watch_controls, text="Dobór do BPTT…", command=self.open_sampling)
        self.sampling_button.pack(side="left", padx=4)
        tk.Label(watch_controls, text="Tempo podglądu (ruchów/s):", bg="#101827", fg="white").pack(side="left", padx=4)
        self.preview_speed = tk.Scale(watch_controls, from_=1, to=60, orient="horizontal",
                                      length=180, command=self.set_preview_speed,
                                      bg="#101827", fg="white", highlightthickness=0)
        self.preview_speed.set(10)
        self.preview_speed.pack(side="left")
        self.preview_max = tk.BooleanVar(value=False)
        tk.Checkbutton(watch_controls, text="MAX", variable=self.preview_max,
                       command=lambda: self.set_preview_speed(self.preview_speed.get()),
                       bg="#101827", fg="white", selectcolor="#101827").pack(side="left", padx=4)
        training_panel = tk.Frame(root, bg="#101827", width=1060, height=76)
        training_panel.pack(pady=(6, 8))
        training_panel.pack_propagate(False)
        self.group_summary_label = tk.Label(training_panel, text="Ostatnie 20 grup: oczekiwanie na wynik",
                                            bg="#101827", fg="#ffe080", font=("Segoe UI", 10))
        self.group_summary_label.pack(side="bottom")
        self.training_status = tk.Label(training_panel, text="AI: RGB 80×60 → CNN → GRU → 4 akcje · ruch ciągły · limit 200",
                                        bg="#101827", fg="#aab8cb", font=("Segoe UI", 10), wraplength=1040)
        self.training_status.pack(fill="both", expand=True)
        root.bind("<KeyPress>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.reset()
        self.poll_callback = root.after(100, self.poll_training)

    @staticmethod
    def validation_quality(result):
        """Q użytkownika: skuteczność rzeczywistych przebiegów walidacji."""
        return result['wins'] / result['maps']

    def update_swatter(self):
        self.canvas.configure(cursor='crosshair' if self.swatter.get() else '')

    def select_bot(self, event=None):
        if self.training_bots and self.live_metadata:
            self.show_training_frame(self.training_bots[0], self.live_metadata)

    def click_bot(self, event):
        if not self.is_watching() or not self.training_bots:
            return
        selected = int(self.selected_bot.get()) - 1
        candidates = [(index, (bot.player.x * CELL-event.x)**2 + (bot.player.y * CELL-event.y)**2)
                      for index, bot in enumerate(self.training_bots) if not (self.swatter.get() and bot.done)]
        candidates = [(index, distance) for index, distance in candidates if distance <= max(9, ROBOT_RADIUS*CELL)**2]
        if not candidates:
            return
        index, _ = min(candidates, key=lambda item: (item[1], item[0] != selected, item[0]))
        if self.swatter.get():
            self.interventions.put((self.live_metadata['episode'], index))
            self.training_status.configure(text=f'Łapka: robot {index+1} · zlecono Bożą karę −20')
        else:
            self.selected_bot.set(str(index+1))
            self.select_bot()

    def load_validation_history(self):
        summary_path = MODEL_PATH.with_name(MODEL_PATH.stem + '.validation.jsonl')
        try:
            if summary_path.exists():
                with summary_path.open(encoding='utf-8') as file:
                    for line in file:
                        try:
                            self.add_validation(json.loads(line), redraw=False)
                        except (ValueError, KeyError, TypeError):
                            continue
            else:
                # Zachowaj także kompletne walidacje ze starszego CSV.
                csv_path = MODEL_PATH.with_name(MODEL_PATH.stem + '.validation.csv')
                if csv_path.exists():
                    group = []
                    with csv_path.open(encoding='utf-8') as file:
                        for row in csv.DictReader(file):
                            if row['map_index'] == '1':
                                group = []
                            group.append(row)
                            if len(group) == int(row['maps']) and row['map_index'] == row['maps']:
                                self.add_validation(dict(timestamp=row['timestamp'], episodes=int(row['episodes']),
                                    kind='periodic' if row['checkpoint'] and int(row['episodes']) % 1000 == 0 else 'manual',
                                    wins=int(row['wins']), maps=int(row['maps']),
                                    score=sum(float(item['score']) for item in group)/len(group)), redraw=False)
                    if self.validation_history:
                        with summary_path.open('w', encoding='utf-8') as file:
                            for entry in self.validation_history:
                                file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except (OSError, ValueError, KeyError):
            self.validation_queue_label.configure(text='Nie udało się wczytać starszego loga walidacji')
        self.draw_validations()

    def add_validation(self, result, redraw=True):
        if not all(key in result for key in ('maps', 'wins', 'score')):
            return
        result = {key: value for key, value in result.items() if key != 'results'}
        self.validation_history.append(result)
        item = self.validation_log.insert('', 'end', values=(result.get('episodes', '—'),
            'co 1000' if result.get('kind') == 'periodic' else 'ręczna',
            f"{result['wins']}/{result['maps']}", f"{self.validation_quality(result):.1%}", f"{result['score']:.1f}"))
        self.validation_log.see(item)
        if redraw:
            self.draw_validations()

    def draw_validations(self):
        automatic = [row for row in self.validation_history if row.get('kind') == 'periodic'][-4:]
        if self.validation_history:
            latest = self.validation_history[-1]
            self.q_label.configure(text=f"Q aktualne: {self.validation_quality(latest):.1%}\n"
                f"{latest['wins']}/{latest['maps']} wygranych · grupa {latest.get('episodes', '—')}")
        else:
            self.q_label.configure(text='Q aktualne: — · brak walidacji')
        text = 'Q średnie: —'
        if automatic:
            average = sum(self.validation_quality(row) for row in automatic) / len(automatic)
            text = f'Q średnie: {average:.1%}'
        text += f'\n{len(automatic)}/4 ostatnich walidacji co 1000'
        self.q_average_label.configure(text=text)
        graph = self.validation_graph
        graph.delete('all')
        entries = self.validation_history[-40:]
        if not entries:
            graph.create_text(155, 45, text='Pierwsza walidacja uzupełni wykres', fill='#aab8cb')
            return
        graph.create_text(3, 8, text='100%', anchor='nw', fill='#aab8cb')
        graph.create_text(3, 60, text='0%', anchor='nw', fill='#aab8cb')
        groups = [row.get('episodes', index) for index, row in enumerate(entries)]
        first, last = min(groups), max(groups)
        points = [(48+(group-first)*250/max(1, last-first), 65-self.validation_quality(row)*53)
                  for group, row in zip(groups, entries)]
        if len(points) > 1:
            graph.create_line(*(v for point in points for v in point), fill='#22d3ee', width=2)
        for (x, y), row in zip(points, entries):
            if row.get('kind') == 'periodic':
                graph.create_oval(x-3, y-3, x+3, y+3, fill='#22d3ee', outline='')
            else:
                graph.create_polygon(x, y-4, x+4, y, x, y+4, x-4, y, fill='#ffe080')
        graph.create_text(48, 82, text=str(first), fill='#aab8cb')
        graph.create_text(298, 82, text=str(last), anchor='e', fill='#aab8cb')

    def open_sampling(self):
        from tkinter import messagebox
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        if hasattr(self, 'sampling_dialog') and self.sampling_dialog.winfo_exists():
            self.sampling_dialog.lift()
            return
        try:
            batch = int(self.batch_count.get())
            if not 1 <= batch <= 1024:
                raise ValueError('Batch musi wynosić od 1 do 1024.')
            settings = self.sampling_settings or load_settings()
            self.sampling_dialog = SamplingDialog(self.root, settings, batch, self.apply_sampling)
        except (ValueError, OSError, KeyError, TypeError) as error:
            messagebox.showerror('Ustawienia BPTT', str(error), parent=self.root)

    def apply_sampling(self, settings):
        self.sampling_settings = settings
        self.training_status.configure(text=(f"BPTT: losowanie {settings['counts']} z grup {settings['percentages']}%."
                                            if settings['enabled'] else 'BPTT: tylko najlepszy robot.'))

    def refresh_start_levels(self):
        try:
            values = ("Kontynuuj", *(str(i) for i in range(1, len(load_plan()) + 1)))
        except (ValueError, OSError):
            values = ("Kontynuuj",)
        self.start_level_input.configure(values=values)
        if self.start_level_choice.get() not in values:
            self.start_level_choice.set("Kontynuuj")

    def open_designer(self):
        from tkinter import messagebox
        if hasattr(self, 'designer') and self.designer.winfo_exists():
            self.designer.lift()
            return
        try:
            plan = load_plan()
        except (ValueError, OSError) as error:
            messagebox.showerror('Nie można wczytać planu', str(error), parent=self.root)
            return
        self.designer = MapDesigner(self.root, plan, COLORS,
            lambda plan: self.training_status.configure(text='Plan zapisany. Następny trening użyje zapisanych poziomów.'),
            on_preview=self.preview_level)

    def preview_level(self, level):
        if self.training_thread is not None and self.training_thread.is_alive():
            raise ValueError('Zatrzymaj trening przed wypróbowaniem poziomu.')
        self.map_seed = random.SystemRandom().randrange(2 ** 31) if level['mode'] == 'random' else None
        self.current_rows = level_rows({'plan': [level], 'level': 0}, self.map_seed)
        self.current_options = level_options(level)
        self.current_level_name = level['name']
        self.reset()
        self.root.lift()

    def reset(self):
        if self.is_watching():
            return
        self.stop_ai()
        self.training_bots = []
        self.live_metadata = {}
        self.episode = Episode(self.current_rows, **self.current_options)
        self.world = self.episode.world
        self.player = self.episode.player
        self.canvas.configure(width=self.world.width * CELL, height=self.world.height * CELL)
        self.map_info.configure(text=f"Seed: {self.map_seed} · bezpieczna droga ≤ 200" if self.map_seed is not None
                                else "Mapa demonstracyjna")
        if self.current_level_name:
            self.map_info.configure(text=self.current_level_name)
        self.message = "Dotrzyj do END. Robot startuje w środku pola; kolizja blokuje ruch."
        self.draw()

    def new_maze(self, seed=None):
        if self.is_watching():
            return
        self.map_seed = random.SystemRandom().randrange(2 ** 31) if seed is None else seed
        self.current_options = {}
        self.current_level_name = ""
        self.current_rows = generate_maze(seed=self.map_seed)
        self.reset()

    def show_demo(self):
        if self.is_watching():
            return
        self.map_seed = None
        self.current_options = {}
        self.current_level_name = ""
        self.current_rows = list(DEMO_MAP)
        self.reset()

    def on_key(self, event):
        key = event.keysym.lower()
        if key == "r":
            self.reset()
            return
        if key == "escape":
            self.close()
            return
        # Edycja liczby epizodów nie może przypadkowo sterować graczem.
        if isinstance(self.root.focus_get(), (tk.Entry, tk.Spinbox)):
            return
        if self.episode.done or self.ai_running or self.is_watching():
            return
        if key in ("a", "left"):
            action = Action.LEFT
        elif key in ("d", "right"):
            action = Action.RIGHT
        elif key in ("s", "down"):
            action = Action.BACKWARD
        elif key in ("w", "up"):
            action = Action.FORWARD
        else:
            return
        self.play_action(action)

    def play_action(self, action):
        result = self.episode.step(action)
        self.message = result.reason or f"{'Krok' if result.moved else 'Obrót'}: {result.duration:.2f} jednostek czasu"
        self.draw()

    def stop_ai(self):
        self.ai_running = False
        if self.ai_callback is not None:
            self.root.after_cancel(self.ai_callback)
            self.ai_callback = None

    def stop_work(self):
        self.stop_ai()
        self.training_stop.set()
        self.watch_training.clear()
        self.clear_preview()
        for queue in (self.validation_requests, self.interventions):
            try:
                while True:
                    queue.get_nowait()
            except Empty:
                pass
        self.watch_button.configure(text="Trenuj i oglądaj")
        self.message = "Zatrzymano AI; jeśli trwa trening, kończy zapisywanie modelu."
        self.draw()

    def start_ai(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        from train import latest_path
        source = latest_path(MODEL_PATH) if latest_path(MODEL_PATH).exists() else MODEL_PATH
        if not source.exists():
            self.training_status.configure(text="Brak modelu. Kliknij „Trenuj dalej” — utworzy nową sieć.")
            return
        try:
            from visual_agent import VisualAgent
            self.agent = VisualAgent.load(source, device=self.device_choice.get())
        except Exception as error:
            self.training_status.configure(text=f"Nie można wczytać AI: {error}")
            return
        self.reset()
        self.episode = Episode(self.current_rows, **self.current_options)
        self.world, self.player = self.episode.world, self.episode.player
        self.agent.reset_memory()
        self.ai_running = True
        self.training_status.configure(text=f"AI gra samodzielnie · model z próby {self.agent.episodes} · bez losowej eksploracji")
        self.ai_tick()

    def ai_tick(self):
        self.ai_callback = None
        if not self.ai_running or self.episode.done:
            self.ai_running = False
            return
        self.play_action(self.agent.act(self.agent.observe(self.episode)))
        if self.episode.done:
            self.ai_running = False
        else:
            self.ai_callback = self.root.after(120, self.ai_tick)

    def is_watching(self):
        return (self.watch_training.is_set() and self.training_thread is not None
                and self.training_thread.is_alive())

    def set_preview_speed(self, value):
        unlimited = hasattr(self, "preview_max") and self.preview_max.get()
        self.preview_delay = 0.0 if unlimited else 1 / max(1, float(value))

    def clear_preview(self):
        try:
            while True:
                self.preview_frames.get_nowait()
        except Empty:
            pass

    def start_watching(self):
        if self.work_kind == "validation":
            return
        if self.training_thread is not None and self.training_thread.is_alive():
            if self.watch_training.is_set():
                self.watch_training.clear()
                self.clear_preview()
                self.watch_button.configure(text="Oglądaj trening")
                self.training_status.configure(text="Podgląd wyłączony — trening działa dalej z pełną prędkością.")
            elif not self.training_stop.is_set():
                self.watch_training.set()
                self.watch_button.configure(text="Ukryj podgląd")
                self.training_status.configure(text="Czekam na ruch treningowy (aktualizacja sieci może chwilę potrwać)…")
            return
        self.start_training(watch=True)

    def publish_training_frame(self, episode, metadata):
        """Wątek treningu wysyła kopię; nigdy nie dotyka obiektów Tkinter."""
        if not self.watch_training.is_set() or self.training_stop.is_set():
            return
        delay = self.preview_delay
        now = monotonic()
        # MAX nie czeka na ekran; kopiuje tylko klatki potrzebne do odświeżania.
        if delay == 0 and now - self.last_preview_time < 0.03:
            return
        self.last_preview_time = now
        frame = deepcopy((episode, metadata))
        self.clear_preview()
        try:
            self.preview_frames.put_nowait(frame)
        except Full:
            pass
        # Ograniczenie tempa dotyczy podglądu, nie czasu wewnątrz gry.
        if delay > 0 and (metadata.get('active', 0) or not episode.done):
            self.training_stop.wait(delay)

    def show_training_frame(self, episode, metadata):
        self.training_bots = metadata.get('bots', [])
        self.live_metadata = metadata
        if self.training_bots:
            self.bot_choice.configure(values=tuple(str(i + 1) for i in range(len(self.training_bots))))
            index = min(max(0, int(self.selected_bot.get()) - 1), len(self.training_bots) - 1)
            self.selected_bot.set(str(index + 1))
            episode = self.training_bots[index]
            metadata = dict(metadata, lane=index + 1)
            if 'bot_actions' in metadata:
                metadata['action'] = metadata['bot_actions'][index]
                metadata['reward'] = metadata['bot_rewards'][index]
        self.episode = episode
        self.world, self.player = episode.world, episode.player
        self.current_rows = list(episode.rows)
        self.current_options = dict(max_time=episode.max_time, move_distance=episode.move_distance, turn_degrees=episode.turn_degrees, start_direction=episode.start_direction)
        self.current_level_name = metadata.get("level_name", "")
        self.map_seed = metadata["map_seed"]
        self.canvas.configure(width=self.world.width * CELL, height=self.world.height * CELL)
        self.map_info.configure(text=f"TRENING · grupa {metadata['episode']} · robot {metadata.get('lane', 1)}/{metadata.get('batch_size', 1)} · seed: {self.map_seed}")
        if "level" in metadata:
            self.map_info.configure(text=f"TRENING · poziom {metadata['level']} {metadata.get('level_name', '')} · grupa {metadata['episode']} · robot {metadata.get('lane', 1)}")
        action = metadata["action"]
        label = "START" if action is None else [f"Przód {episode.move_distance:g} m", f"Lewo {episode.turn_degrees:g}°", f"Prawo {episode.turn_degrees:g}°", f"Tył {episode.move_distance:g} m"][action]
        self.message = (f"{label} · nagroda: {metadata['reward']:+.3f} · "
                        f"eksploracja: {metadata['epsilon']:.0%} · krok {episode.steps}")
        if episode.done:
            self.message += f" · {episode.outcome}"
        self.training_status.configure(text=f"Trening na żywo · grupa {metadata['episode']} "
            f"(+{metadata['session_episode']}) · robot {metadata.get('lane', 1)}: krok {episode.steps} · "
            f"łącznie {metadata.get('total_steps', episode.steps)} ruchów · {metadata.get('device', '')} · aktywnych {metadata.get('active', 1)}/{metadata.get('batch_size', 1)} · "
            f"{metadata.get('steps_per_second', 0):.0f} kroków/s")
        self.draw()

    def start_training(self, watch=False):
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        try:
            count = int(self.episode_count.get())
            batch_size = int(self.batch_count.get())
            device = self.device_choice.get()
            plan = load_plan()
            sampling = self.sampling_settings or load_settings()
            selection = None
            if sampling['enabled']:
                validate_sampling(batch_size, sampling['percentages'], sampling['counts'])
                selection = {key: list(sampling[key]) for key in ('percentages', 'counts')}
            choice = self.start_level_choice.get()
            start_level = None if choice == "Kontynuuj" else int(choice)
            if start_level is not None and not 1 <= start_level <= len(plan):
                raise ValueError(f"Poziom startowy musi wynosić od 1 do {len(plan)}.")
            if not 1 <= count <= 1_000_000 or not 1 <= batch_size <= 1024:
                raise ValueError
        except (ValueError, OSError) as error:
            self.training_status.configure(text=f"{error} · Podaj 1–1 000 000 grup oraz batch 1–1024.")
            return
        if hasattr(self, 'sampling_dialog') and self.sampling_dialog.winfo_exists():
            self.sampling_dialog.destroy()
        self.sampling_button.configure(state="disabled")
        self.stop_ai()
        self.training_stop = Event()
        self.clear_preview()
        self.training_bots = []
        self.live_metadata = {}
        try:
            while True:
                self.interventions.get_nowait()
        except Empty:
            pass
        if watch:
            self.watch_training.set()
        else:
            self.watch_training.clear()
        self.watch_button.configure(text="Ukryj podgląd" if watch else "Oglądaj trening")
        self.work_kind = "training"
        self.start_level_input.configure(state="disabled")
        self.start_level_choice.set("Kontynuuj")
        self.batch_input.configure(state="disabled")
        self.device_input.configure(state="disabled")
        self.validation_button.configure(state="normal")
        self.train_button.configure(state="disabled")
        self.ai_button.configure(state="disabled")
        self.training_status.configure(text="Wczytuję zapis nauki i uruchamiam trening według planu poziomów…")

        def worker():
            try:
                from train import train
                report = train(count, model_path=MODEL_PATH, batch_size=batch_size, device=device,
                               on_status=lambda message: self.events.put(("phase", message)),
                               progress=lambda row: self.events.put(("progress", row)),
                               stop=self.training_stop, on_step=self.publish_training_frame, plan=plan, start_level=start_level, selection=selection,
                               validation_requests=self.validation_requests, interventions=self.interventions,
                               on_validation=lambda result: self.events.put(('validation_result', result)))
                self.events.put(("finished", report))
            except Exception as error:
                self.events.put(("error", str(error)))

        self.training_thread = Thread(target=worker, daemon=True)
        self.training_thread.start()

    def start_validation(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            if not self.training_stop.is_set():
                self.validation_requests.put('manual')
                self.validation_queue_label.configure(text=f'Kolejka walidacji: {self.validation_requests.qsize()} · po bieżącej grupie / ocenie')
            return
        from train import latest_path
        source = latest_path(MODEL_PATH) if latest_path(MODEL_PATH).exists() else MODEL_PATH
        if not source.exists():
            self.training_status.configure(text="Brak modelu do walidacji. Najpierw uruchom trening.")
            return
        self.stop_ai()
        self.training_stop = Event()
        self.work_kind = "validation"
        device = self.device_choice.get()
        self.watch_training.clear()
        self.clear_preview()
        for button in (self.train_button, self.ai_button, self.watch_button):
            button.configure(state="disabled")
        self.training_status.configure(text="Walidacja najnowszego modelu na 100 mapach 11×11… Stop przerywa ocenę.")

        def worker():
            try:
                from visual_agent import VisualAgent
                from train import run_validation
                agent = VisualAgent.load(source, device=device)
                while True:
                    result = run_validation(agent, MODEL_PATH, stop=self.training_stop, checkpoint_path=source)
                    self.events.put(('validation_result', result))
                    if self.training_stop.is_set():
                        raise InterruptedError
                    try:
                        self.validation_requests.get_nowait()
                    except Empty:
                        break
                self.events.put(("validated", result))
            except InterruptedError:
                self.events.put(("validation_stopped", None))
            except Exception as error:
                self.events.put(("error", str(error)))

        self.training_thread = Thread(target=worker, daemon=True)
        self.training_thread.start()

    def poll_training(self):
        try:
            self.resource_label.configure(text=format_resources(self.resource_monitor.samples.get_nowait()))
        except Empty:
            pass
        try:
            episode, metadata = self.preview_frames.get_nowait()
            if self.watch_training.is_set() and not self.training_stop.is_set():
                self.show_training_frame(episode, metadata)
        except Empty:
            pass
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "phase":
                    self.training_status.configure(text=data)
                elif kind == 'validation_result':
                    self.add_validation(data)
                elif kind == "progress":
                    self.group_summary_label.configure(text=f"Ostatnie 20 grup: {data.get('recent_group_wins', 0)}/"
                        f"{data.get('recent_group_count', 0)} wygranych ({data['success_rate']:.0%})")
                    if 'level' in data:
                        self.group_summary_label.configure(text=("Plan ukończony!" if data['completed'] else
                            f"Poziom {data['next_level']}/{data['level_count']} · seria wygranych: {data['level_wins']}/{data['required']} z rzędu"))
                    promotion = (f" · awans → {data['next_width']}×{data['next_height']}"
                                 if data.get("next_width", data["width"]) != data["width"]
                                 or data.get("next_height", data["height"]) != data["height"] else "")
                    self.training_status.configure(text=f"Trening: grupa {data['episode']} (+{data['session_episode']}) · "
                        f"mapa {data['width']}×{data['height']} · "
                        f"BPTT: {data.get('bptt_count', 1)} przebiegów · najlepszy {data.get('winner', 1)}/{data.get('batch_size', 1)} · nagrody: {data['reward']:+.2f} · "
                        f"{data.get('steps_per_second', 0):.0f} kroków/s · BPTT {data.get('learn_seconds', 0):.2f} s{promotion}")
                else:
                    self.watch_training.clear()
                    self.clear_preview()
                    self.watch_button.configure(text="Trenuj i oglądaj")
                    if kind == "error":
                        self.training_status.configure(text=f"Błąd: {data}")
                    elif kind == "validated":
                        self.training_status.configure(text=f"Walidacja: {data['wins']}/{data['maps']} wygranych · "
                            f"średni wynik: {data['score']:.1f}")
                    elif kind == "validation_stopped":
                        self.training_status.configure(text="Walidacja zatrzymana.")
                    elif data.get("error"):
                        self.training_status.configure(text=data["error"])
                    else:
                        self.training_status.configure(text=f"Zapisano naukę (+{data['training_episodes']} grup, {data.get('training_rollouts', data['training_episodes'])} przebiegów). "
                            + ("Plan poziomów ukończony!" if data.get("curriculum", {}).get("completed") else "Możesz trenować dalej, uruchomić AI lub kliknąć Walidacja."))
        except Empty:
            pass
        if self.training_thread is not None and not self.training_thread.is_alive():
            self.training_thread = None
            self.work_kind = None
            self.sampling_button.configure(state="normal")
            self.start_level_input.configure(state="readonly")
            self.batch_input.configure(state="normal")
            self.device_input.configure(state="normal")
            self.watch_training.clear()
            self.watch_button.configure(text="Trenuj i oglądaj")
            for button in (self.train_button, self.ai_button, self.validation_button, self.watch_button):
                button.configure(state="normal")
            # Kliknięcie przy samym końcu pracy też uruchamia ocenę.
            if not self.training_stop.is_set() and not self.validation_requests.empty():
                self.validation_requests.get_nowait()
                self.start_validation()
        self.validation_queue_label.configure(text=f'Kolejka walidacji: {self.validation_requests.qsize()}')
        self.poll_callback = self.root.after(30, self.poll_training)

    def close(self):
        self.resource_monitor.close()
        self.stop_ai()
        self.training_stop.set()
        self.root.after_cancel(self.poll_callback)
        self.root.destroy()

    def draw(self):
        self.movement_label.configure(text=f"Pole: 1×1 m · robot: Ø20 cm · krok: {self.player.move_distance:g} m · obrót: {self.player.turn_degrees:g}°")
        show_camera = self.camera_enabled.get() and (self.work_kind != "training" or self.is_watching())
        if show_camera:
            if self._preview_world is not self.world:
                self._preview_world = self.world
                self._preview_camera = TensorCamera(self.world, "cpu")
            self.camera_frame = self._preview_camera.render([camera_pose(self.player)])[0].numpy()
            self.camera_image = tk.PhotoImage(data=camera_ppm(self.camera_frame), format="PPM").zoom(4)
            self.camera_view.configure(image=self.camera_image, text="")
        else:
            self.camera_frame = None
            self.camera_image = None
            self.camera_view.configure(image="", text="Podgląd kamery wyłączony")
        canvas = self.canvas
        canvas.delete("all")
        for y in range(self.world.height):
            for x in range(self.world.width):
                terrain = self.world.tile_at(x, y).terrain
                left, top = x * CELL, y * CELL
                canvas.create_rectangle(left + 1, top + 1, left + CELL - 1, top + CELL - 1,
                                        fill=COLORS[terrain][0], outline="#253349")
                if terrain in (Terrain.START, Terrain.END):
                    canvas.create_text(left + CELL / 2, top + 9,
                                       text="START" if terrain is Terrain.START else "END",
                                       fill="#101827", font=("Segoe UI", 8, "bold"))
                elif terrain in (Terrain.WATER, Terrain.SWAMP, Terrain.FIRE, Terrain.PAVEMENT):
                    canvas.create_text(left + CELL / 2, top + CELL / 2,
                                       text={Terrain.WATER: "≈", Terrain.SWAMP: "≋",
                                             Terrain.FIRE: "F", Terrain.PAVEMENT: "▦"}[terrain],
                                       fill="#172b3a", font=("Segoe UI", 20, "bold"))
        cx, cy = self.player.x * CELL, self.player.y * CELL
        dx, dy = self.player.direction_vector
        radius = ROBOT_RADIUS * CELL
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                           fill="#ffe080" if self.player.alive else "#ff6666", outline="white", tags="robot")
        canvas.create_line(cx, cy, cx + dx * radius, cy + dy * radius,
                           fill="#15243b", width=2, tags="heading")
        if self.training_bots:
            canvas.delete('robot')
            canvas.delete('heading')
            colors = ('#ffe080', '#22d3ee', '#f472b6', '#a3e635', '#c4b5fd', '#fb923c')
            selected = int(self.selected_bot.get()) - 1
            for index, bot in enumerate(self.training_bots):
                player = bot.player
                x, y = player.x * CELL, player.y * CELL
                color = '#ef4444' if bot.swatted else ('#64748b' if bot.done else colors[index % len(colors)])
                canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color,
                                   outline='white' if index == selected else '#15243b',
                                   width=2 if index == selected else 1, tags=f'bot-{index}')
                dx, dy = player.direction_vector
                canvas.create_line(x, y, x+dx*radius, y+dy*radius, fill='#15243b', width=2)
                if index == selected:
                    canvas.create_text(x, y-12, text=str(index+1), fill='white',
                                        font=('Segoe UI', 9, 'bold'))
        if self.episode.done and not self.is_watching():
            width, height = self.world.width * CELL, self.world.height * CELL
            margin = min(80, width * 0.08)
            canvas.create_rectangle(margin, height / 2 - 60, width - margin, height / 2 + 60,
                                    fill="#101827", outline="#ffe080", width=2)
            canvas.create_text(width / 2, height / 2 - 15,
                               text=("WYGRANA!" if self.player.won else
                                     "KONIEC CZASU" if self.episode.timed_out else "PRZEGRANA"),
                               fill="white", font=("Segoe UI", 16 if width < 400 else 20, "bold"),
                               width=width - 2 * margin - 12)
            canvas.create_text(width / 2, height / 2 + 24,
                               text="Naciśnij R, aby zacząć od nowa",
                               fill="#cbd5e1", font=("Segoe UI", 11), width=width - 2 * margin - 12)
        p = self.player
        hazard = f"Woda: {p.water_distance:.1f}/3 m | Ogień: {p.fire_distance:.1f}/2 m"
        if p.burning_distance is not None:
            hazard += f" | Podpalenie: {p.burning_distance:.1f}/3 m"
        self.status.configure(text=f"Czas: {p.elapsed_time:.2f}/{self.episode.max_time:g} | Obrażenia: {p.damage:.1f} | "
                              f"Wynik: {self.episode.score:.1f} | {hazard}\n"
                              f"Nagrody: {self.episode.total_reward:+.2f} | Eksploracja: {self.episode.reward_totals.get('new_tile', 0):+.2f} | "
                              f"Kary za kolizje: {self.episode.reward_totals.get('collision', 0):+.2f}\n{self.message}\n"
                              f"Pozycja: ({p.x:.2f}, {p.y:.2f}) m | Kierunek: {p.heading:g}° | "
                              f"Odwiedzone pola: {len(self.episode.visited)} | Kolizje: {self.episode.collisions}")
