"""Klikany edytor planu map treningowych."""
from copy import deepcopy
from math import isfinite
import tkinter as tk
from tkinter import messagebox, ttk
from curriculum import SIZES, blank_map, save_plan, validate_plan, validate_fields, DEFAULT_FIELDS
from world import Terrain
from player import validate_movement
from maps import validate_size


DIRECTIONS = {'Losowy': 'random', 'Prawo →': 'right', 'Lewo ←': 'left', 'Góra ↑': 'up', 'Dół ↓': 'down'}


class MapDesigner(tk.Toplevel):
    def __init__(self, parent, plan, colors, on_save, on_preview=None):
        super().__init__(parent)
        self.title('Projektant map i poziomów')
        self.plan, self.colors, self.on_save = deepcopy(plan), colors, on_save
        self.on_preview = on_preview
        self.index = 0
        self.pola = tk.StringVar(value=DEFAULT_FIELDS)
        self.brush = tk.StringVar(value='.')
        self.mode = tk.StringVar()
        self.size = tk.StringVar()
        self.required = tk.StringVar()
        self.max_time = tk.StringVar()
        self.name = tk.StringVar()
        self.move_distance = tk.StringVar()
        self.turn_degrees = tk.StringVar()
        self.start_direction = tk.StringVar(value='Prawo →')
        left = ttk.Frame(self, padding=12)
        left.pack(side='left', fill='y')
        ttk.Label(left, text='Kolejność poziomów').pack()
        self.levels = tk.Listbox(left, width=39, height=18, exportselection=False)
        self.levels.pack(fill='y', expand=True, pady=8)
        scroll = ttk.Scrollbar(left, orient='horizontal', command=self.levels.xview)
        scroll.pack(fill='x')
        self.levels.configure(xscrollcommand=scroll.set)
        self.levels.bind('<<ListboxSelect>>', self.select)
        ttk.Button(left, text='Dodaj poziom', command=self.add).pack(fill='x')
        ttk.Button(left, text='Usuń poziom', command=self.remove).pack(fill='x', pady=4)
        right = ttk.Frame(self, padding=12)
        right.pack(side='left')
        controls = ttk.Frame(right)
        controls.pack(fill='x')
        ttk.Label(controls, text='Mapa:').grid(row=0, column=0)
        for column, (label, value) in enumerate((('Losowa', 'random'), ('Moja', 'custom')), 1):
            ttk.Radiobutton(controls, text=label, variable=self.mode, value=value,
                            command=self.draw).grid(row=0, column=column)
        ttk.Label(controls, text='Rozmiar:').grid(row=1, column=0)
        menu = ttk.Combobox(controls, textvariable=self.size, values=SIZES, width=6)
        menu.grid(row=1, column=1)
        menu.bind('<<ComboboxSelected>>', self.resize_map)
        menu.bind('<Return>', self.resize_map)
        menu.bind('<FocusOut>', self.resize_map)
        ttk.Label(controls, text='Pola losowe · wymagane S E #').grid(row=1, column=2)
        ttk.Entry(controls, textvariable=self.pola, width=20).grid(row=1, column=3)
        ttk.Label(controls, text='Zaliczenia z rzędu:').grid(row=2, column=0)
        ttk.Spinbox(controls, from_=1, to=1000000, textvariable=self.required, width=9).grid(row=2, column=1)
        ttk.Label(controls, text='Limit czasu mapy (s):').grid(row=3, column=0)
        ttk.Spinbox(controls, from_=2, to=1000000, increment=10, textvariable=self.max_time, width=9).grid(row=3, column=1)
        ttk.Label(controls, text='Krok robota (m):').grid(row=4, column=0)
        ttk.Spinbox(controls, from_=.01, to=1, increment=.1, textvariable=self.move_distance, width=9).grid(row=4, column=1)
        ttk.Label(controls, text='Kąt obrotu (°):').grid(row=4, column=2)
        ttk.Spinbox(controls, from_=1, to=180, increment=5, textvariable=self.turn_degrees, width=9).grid(row=4, column=3)
        ttk.Label(controls, text='Nazwa / opis:').grid(row=5, column=0)
        ttk.Entry(controls, textvariable=self.name, width=52).grid(row=5, column=1, columnspan=3, sticky='ew')
        ttk.Label(controls, text='Kierunek startowy:').grid(row=6, column=0)
        ttk.Combobox(controls, textvariable=self.start_direction, values=list(DIRECTIONS),
                     state='readonly', width=14).grid(row=6, column=1)
        ttk.Label(controls, text='Losowy: osobno dla każdego bota, 4 kierunki.').grid(row=6, column=2, columnspan=2)
        palette = ttk.Frame(right)
        palette.pack(pady=8)
        for i, (terrain, (color, label)) in enumerate(colors.items()):
            tk.Radiobutton(palette, text=f"{label} [{terrain.value}]", variable=self.brush, value=terrain.value,
                           indicatoron=False, bg=color, selectcolor=color, width=18).grid(row=i//4, column=i%4)
        self.canvas = tk.Canvas(right, width=440, height=440, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.paint)
        self.canvas.bind('<B1-Motion>', self.paint)
        ttk.Label(right, text='Wybierz „Moja”, wybierz teren i maluj kliknięciem lub przeciąganiem.\n'
                  'START i END są przenoszone. Losowa: nowa mapa w każdej grupie.\n'
                  'Jedna wygrana grupa = jedno zaliczenie; porażka przerywa serię.\n'
                  'Zmiana planu zaczyna poziom 1, zachowując wyuczone wagi.').pack(pady=8)
        ttk.Button(right, text='Zapisz plan', command=self.save).pack()
        if on_preview is not None:
            ttk.Button(right, text='Wypróbuj poziom', command=self.preview).pack(pady=4)
        self.load()

    def refresh(self):
        self.levels.delete(0, 'end')
        for i, level in enumerate(self.plan, 1):
            kind = 'losowa' if level['mode'] == 'random' else 'moja'
            self.levels.insert('end', f"{i}. {level.get('name', f'Poziom {i}')} · {kind} {level['size']}×{level['size']} · {level['required']} razy · {level.get('max_time', 200):g} s · {level.get('move_distance', .1):g} m / {level.get('turn_degrees', 10):g}°")
        self.levels.selection_set(self.index)

    def commit(self):
        if not self.resize_map():
            return False
        try:
            required = int(self.required.get())
            if not 1 <= required <= 1000000:
                raise ValueError
        except ValueError:
            messagebox.showerror('Liczba zaliczeń', 'Podaj liczbę całkowitą 1–1 000 000.', parent=self)
            return False
        try:
            max_time = float(self.max_time.get().replace(',', '.'))
            if not isfinite(max_time) or not 2 <= max_time <= 1000000:
                raise ValueError
        except ValueError:
            messagebox.showerror('Limit czasu', 'Podaj czas od 2 do 1 000 000 sekund.', parent=self)
            return False
        try:
            distance = float(self.move_distance.get().replace(',', '.'))
            angle = float(self.turn_degrees.get().replace(',', '.'))
            validate_movement(distance, angle)
            name = self.name.get().strip()
            if not 1 <= len(name) <= 120:
                raise ValueError('Nazwa / opis: wpisz od 1 do 120 znaków.')
        except ValueError as error:
            messagebox.showerror('Ustawienia poziomu', str(error), parent=self)
            return False
        try:
            fields = validate_fields(self.pola.get())
        except ValueError as error:
            messagebox.showerror('Pola dostępne', str(error), parent=self)
            return False
        self.plan[self.index].update(mode=self.mode.get(), required=required, max_time=max_time,
                                     name=name, move_distance=distance, turn_degrees=angle,
                                     start_direction=DIRECTIONS[self.start_direction.get()], pola=fields)
        return True

    def load(self):
        level = self.plan[self.index]
        self.mode.set(level['mode'])
        self.size.set(str(level['size']))
        self.required.set(str(level['required']))
        self.max_time.set(f"{level.get('max_time', 200):g}")
        self.name.set(level.get('name', f'Poziom {self.index+1}'))
        self.move_distance.set(f"{level.get('move_distance', .1):g}")
        self.turn_degrees.set(f"{level.get('turn_degrees', 10):g}")
        self.start_direction.set(next(label for label, value in DIRECTIONS.items() if value == level.get('start_direction', 'right')))
        self.pola.set(level.get('pola', DEFAULT_FIELDS))
        self.refresh()
        self.draw()

    def select(self, event=None):
        selection = self.levels.curselection()
        if not selection or selection[0] == self.index:
            return
        target = selection[0]
        if self.commit():
            self.index = target
            self.load()
        else:
            self.refresh()

    def resize_map(self, event=None):
        level = self.plan[self.index]
        try:
            size = validate_size(int(self.size.get()))
        except ValueError as error:
            messagebox.showerror('Rozmiar mapy', str(error), parent=self)
            self.size.set(str(level['size']))
            return False
        if size == level['size']:
            return True
        if not messagebox.askyesno('Zmiana rozmiaru', 'Zmienić rozmiar i wyczyścić rysunek tego poziomu?', parent=self):
            self.size.set(str(level['size']))
            return False
        level.update(size=size, rows=blank_map(size))
        self.draw()
        return True

    def paint(self, event):
        if self.mode.get() != 'custom':
            return
        level = self.plan[self.index]
        cell = 440 / level['size']
        x, y = int(event.x / cell), int(event.y / cell)
        if not (0 <= x < level['size'] and 0 <= y < level['size']):
            return
        rows = [list(row) for row in level['rows']]
        symbol = self.brush.get()
        if symbol in 'SE':
            rows = [['.' if tile == symbol else tile for tile in row] for row in rows]
        rows[y][x] = symbol
        level['rows'] = [''.join(row) for row in rows]
        self.draw()

    def draw(self):
        self.canvas.delete('all')
        if self.mode.get() == 'random':
            self.canvas.create_text(220, 220, text='MAPA LOSOWA\nNowy układ przy każdej próbie', justify='center', font=('Segoe UI', 16))
            return
        level = self.plan[self.index]
        cell = 440 / level['size']
        if cell < 12:
            from map_view import terrain_ppm
            self.map_image = tk.PhotoImage(data=terrain_ppm(level['rows'], cell,
                {terrain.value: color for terrain, (color, _) in self.colors.items()}), format='PPM')
            self.canvas.create_image(0, 0, image=self.map_image, anchor='nw')
            for y, row in enumerate(level['rows']):
                for x, tile in enumerate(row):
                    if tile in 'SE':
                        self.canvas.create_text((x+.5)*cell, (y+.5)*cell, text=tile,
                                                fill='white', font=('Segoe UI', 9, 'bold'))
            return
        for y, row in enumerate(level['rows']):
            for x, tile in enumerate(row):
                self.canvas.create_rectangle(x*cell, y*cell, (x+1)*cell, (y+1)*cell,
                                             fill=self.colors[Terrain(tile)][0], outline='#253349')
                if tile in 'SE':
                    self.canvas.create_text((x+.5)*cell, (y+.5)*cell, text=tile, font=('Segoe UI', 12, 'bold'))

    def add(self):
        if self.commit():
            self.plan.append(dict(name=f'Poziom {len(self.plan)+1}', start_direction='right', move_distance=.1, turn_degrees=10., mode='random', size=5, required=20, max_time=200.0, pola=DEFAULT_FIELDS, rows=blank_map(5)))
            self.index = len(self.plan)-1
            self.load()

    def remove(self):
        if len(self.plan) > 1 and messagebox.askyesno('Usuń poziom', f'Usunąć poziom {self.index+1}?', parent=self):
            self.plan.pop(self.index)
            self.index = min(self.index, len(self.plan)-1)
            self.load()

    def preview(self):
        if not self.commit():
            return
        try:
            level = validate_plan([self.plan[self.index]])[0]
            self.on_preview(level)
        except ValueError as error:
            messagebox.showerror('Podgląd poziomu', str(error), parent=self)

    def save(self):
        if not self.commit():
            return
        try:
            save_plan(self.plan)
        except (ValueError, OSError) as error:
            messagebox.showerror('Nie zapisano planu', str(error), parent=self)
            return
        self.on_save(deepcopy(self.plan))
        self.refresh()
        messagebox.showinfo('Plan zapisany', 'Plan zostanie użyty przy następnym uruchomieniu treningu.', parent=self)
