"""Ustawienia wyboru przebiegów do wspólnego kroku BPTT."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from sampling import validate_sampling

from config.config import SETTINGS_PATH


def load_settings():
    if not SETTINGS_PATH.exists():
        return dict(enabled=False, percentages=[10, 20, 70], counts=[3, 2, 3])
    settings = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
    if not isinstance(settings, dict) or type(settings.get('enabled')) is not bool:
        raise ValueError('Nieprawidłowe ustawienia BPTT')
    if not isinstance(settings.get('percentages'), list) or not isinstance(settings.get('counts'), list):
        raise ValueError('Nieprawidłowy podział lub liczby przebiegów BPTT')
    validate_sampling(1024, settings['percentages'], settings['counts'])
    return settings


class SamplingDialog(tk.Toplevel):
    def __init__(self, parent, settings, batch_size, apply):
        super().__init__(parent)
        self.title('Wybór przebiegów do BPTT')
        self.batch_size, self.apply = batch_size, apply
        self.enabled = tk.BooleanVar(value=settings['enabled'])
        self.percentages = [tk.StringVar(value=str(v)) for v in settings['percentages']]
        self.counts = [tk.StringVar(value=str(v)) for v in settings['counts']]
        panel = ttk.Frame(self, padding=16)
        panel.pack()
        ttk.Checkbutton(panel, text='Losuj z trzech grup (wyłączone: tylko najlepszy robot)', variable=self.enabled).grid(row=0, column=0, columnspan=3, pady=8)
        for col, title in enumerate(('Część rankingu', 'Podział (%)', 'Ile do BPTT')):
            ttk.Label(panel, text=title).grid(row=1, column=col, padx=10)
        for i, name in enumerate(('1. Najlepsi', '2. Średni', '3. Pozostali')):
            ttk.Label(panel, text=name).grid(row=i+2, column=0)
            ttk.Spinbox(panel, from_=0, to=100, textvariable=self.percentages[i], width=8).grid(row=i+2, column=1, pady=5)
            ttk.Spinbox(panel, from_=0, to=1024, textvariable=self.counts[i], width=8).grid(row=i+2, column=2)
        self.summary = ttk.Label(panel, wraplength=490, justify='left')
        self.summary.grid(row=5, column=0, columnspan=3, pady=12)
        ttk.Label(panel, text='Ranking: ukończenie → nagrody → wynik → mniej kroków.\n'
                  'Każdy wybrany przebieg: własna pamięć GRU od zera.\n'
                  'Średnia strat przebiegów → jedna aktualizacja wspólnych wag.\n'
                  'Losowanie bez powtórzeń. Procenty muszą dawać 100%.', justify='left').grid(row=6, column=0, columnspan=3)
        ttk.Button(panel, text='Zapisz ustawienia', command=self.save).grid(row=7, column=0, columnspan=3, pady=12)
        for variable in self.percentages + self.counts:
            variable.trace_add('write', self.preview)
        self.preview()

    def values(self):
        return [int(v.get()) for v in self.percentages], [int(v.get()) for v in self.counts]

    def preview(self, *args):
        try:
            percentages, counts = self.values()
            sizes = validate_sampling(self.batch_size, percentages, counts)
            shares = ' / '.join(f'{100*n/sum(counts):.1f}%' for n in counts)
            self.summary.configure(text=f'Batch {self.batch_size}: grupy {sizes[0]} / {sizes[1]} / {sizes[2]} robotów.\n'
                                   f'BPTT: {sum(counts)} przebiegów. Udziały w stracie: {shares}.')
        except ValueError as error:
            self.summary.configure(text=str(error))

    def save(self):
        try:
            percentages, counts = self.values()
            validate_sampling(self.batch_size if self.enabled.get() else 1024, percentages, counts)
            settings = dict(enabled=self.enabled.get(), percentages=percentages, counts=counts)
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = SETTINGS_PATH.with_suffix('.tmp')
            temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
            temporary.replace(SETTINGS_PATH)
        except (ValueError, OSError) as error:
            messagebox.showerror('Ustawienia BPTT', str(error), parent=self)
            return
        self.apply(settings)
        self.destroy()
