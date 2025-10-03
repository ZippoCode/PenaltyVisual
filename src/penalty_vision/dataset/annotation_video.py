import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional
import subprocess


class PenaltyAnnotator:
    """Annotatore dettagliato per clips di rigori"""

    def __init__(self, video_dir: str = "dataset/raw_videos",
                 csv_file: str = "dataset/annotations.csv"):
        self.video_dir = Path(video_dir)
        self.csv_file = Path(csv_file)
        self.annotations = []

        # Carica annotazioni esistenti
        self._load_existing()

    def _load_existing(self):
        """Carica annotazioni già fatte"""
        if self.csv_file.exists():
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                self.annotations = list(reader)
            print(f"✓ Caricate {len(self.annotations)} annotazioni esistenti")
        else:
            # Crea CSV con header
            self.csv_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'video_file', 'piede', 'lato', 'altezza', 'dentro_fuori', 'parato',
                    'angolo_camera', 'visibilita_giocatore', 'velocita_rincorsa', 'fake'
                ])
                writer.writeheader()
            print("✓ Creato nuovo file annotations.csv")

    def _get_annotated_files(self) -> set:
        """Ottiene lista file già annotati"""
        return {ann['video_file'] for ann in self.annotations}

    def _play_video(self, video_path: Path):
        """Apre video con player di sistema"""
        try:
            if sys.platform == 'darwin':  # Mac
                # Prova IINA, poi VLC, poi default
                for app in ['vlc', 'open']:
                    try:
                        subprocess.Popen([app, str(video_path)],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                        return
                    except FileNotFoundError:
                        continue
            elif sys.platform == 'linux':
                subprocess.Popen(['xdg-open', str(video_path)])
            elif sys.platform == 'win32':
                subprocess.Popen(['start', str(video_path)], shell=True)
        except Exception as e:
            print(f"⚠ Impossibile aprire video automaticamente: {e}")
            print(f"Apri manualmente: {video_path}")

    def _get_input(self, prompt: str, valid_options: List[str]) -> str:
        """Ottiene input validato dall'utente"""
        while True:
            response = input(prompt).strip().lower()
            if response in valid_options:
                return response
            print(f"❌ Opzione non valida. Scegli tra: {', '.join(valid_options)}")

    def _print_grid(self):
        """Mostra griglia 3x3 posizioni"""
        print("""
        ╔═══════════════════════════════════╗
        ║         GRIGLIA PORTA             ║
        ╠═════════╦═════════╦═══════════════╣
        ║    1    ║    2    ║       3       ║
        ║ sinistra║ centro  ║     destra    ║
        ║  alto   ║  alto   ║      alto     ║
        ╠═════════╬═════════╬═══════════════╣
        ║    4    ║    5    ║       6       ║
        ║ sinistra║ centro  ║     destra    ║
        ║ centrale║ centrale║    centrale   ║
        ╠═════════╬═════════╬═══════════════╣
        ║    7    ║    8    ║       9       ║
        ║ sinistra║ centro  ║     destra    ║
        ║  basso  ║  basso  ║      basso    ║
        ╚═════════╩═════════╩═══════════════╝
        
        (Prospettiva: dalla telecamera)
        """)

    def annotate_video(self, video_file: str) -> Optional[Dict]:
        """Annota singolo video con schema dettagliato"""
        print(f"\n{'=' * 60}")
        print(f"📹 Video: {video_file}")
        print('=' * 60)

        # 1. Angolo Telecamera (PRIMA DI TUTTO)
        print("\n🎥 ANGOLO TELECAMERA?")
        print("   [l] Lateral (laterale classico) - IDEALE")
        print("   [d] Diagonal (diagonale)")
        print("   [b] Behind goal (dietro porta)")
        print("   [f] Frontal (frontale)")
        angolo = self._get_input("   → ", ['l', 'd', 'b', 'f'])
        angolo_map = {'l': 'lateral', 'd': 'diagonal', 'b': 'behind_goal', 'f': 'frontal'}
        angolo_camera = angolo_map[angolo]

        # 2. Visibilità Giocatore
        print("\n👤 VISIBILITÀ GIOCATORE (corpo intero)?")
        print("   [f] Full (completamente visibile)")
        print("   [p] Partial (parzialmente occluso)")
        print("   [o] Poor (molto occluso)")
        vis = self._get_input("   → ", ['f', 'p', 'o'])
        vis_map = {'f': 'full', 'p': 'partial', 'o': 'poor'}
        visibilita = vis_map[vis]

        # 3. Piede
        print("\n🦶 PIEDE CALCIANTE?")
        piede = self._get_input("   [r] Destro  [l] Sinistro  → ", ['r', 'l'])
        piede = "right" if piede == 'r' else "left"

        # 4. Velocità Rincorsa
        print("\n⚡ VELOCITÀ RINCORSA?")
        print("   [s] Slow (lenta/corta)")
        print("   [m] Medium (normale)")
        print("   [f] Fast (veloce/lunga)")
        vel = self._get_input("   → ", ['s', 'm', 'f'])
        vel_map = {'s': 'slow', 'm': 'medium', 'f': 'fast'}
        velocita = vel_map[vel]

        # 5. Fake/Deception
        print("\n🎭 FINTA/CAMBIO DIREZIONE?")
        fake = self._get_input("   [y] Yes (fa finta)  [n] No (tiro diretto)  → ", ['y', 'n'])
        fake = "yes" if fake == 'y' else "no"

        # 6. Dentro/Fuori
        print("\n⚽ ESITO DEL TIRO?")
        dentro_fuori = self._get_input("   [d] Dentro (in porta)  [f] Fuori (palo/traversa/out)  → ", ['d', 'f'])
        dentro_fuori = "dentro" if dentro_fuori == 'd' else "fuori"

        # 7. Parato (solo se dentro)
        parato = "n/a"
        if dentro_fuori == "dentro":
            print("\n🧤 PARATO DAL PORTIERE?")
            parato_input = self._get_input("   [p] Parato  [g] Gol  → ", ['p', 'g'])
            parato = "parato" if parato_input == 'p' else "gol"

        # 8. Posizione
        if dentro_fuori == "fuori":
            print("\n📍 DIREZIONE TENTATA (anche se fuori)?")
            print("   (Dove sarebbe andato se non usciva)")
        else:
            print("\n📍 POSIZIONE IN PORTA?")

        self._print_grid()

        grid_pos = self._get_input("   Numero [1-9]  → ",
                                   ['1', '2', '3', '4', '5', '6', '7', '8', '9'])

        # Mappa numero → (lato, altezza)
        grid_map = {
            '1': ('left', 'high'),
            '2': ('center', 'high'),
            '3': ('right', 'high'),
            '4': ('left', 'mid'),
            '5': ('center', 'mid'),
            '6': ('right', 'mid'),
            '7': ('left', 'low'),
            '8': ('center', 'low'),
            '9': ('right', 'low'),
        }

        lato, altezza = grid_map[grid_pos]

        annotation = {
            'video_file': video_file,
            'piede': piede,
            'lato': lato,
            'altezza': altezza,
            'dentro_fuori': dentro_fuori,
            'parato': parato,
            'angolo_camera': angolo_camera,
            'visibilita_giocatore': visibilita,
            'velocita_rincorsa': velocita,
            'fake': fake
        }

        # Conferma con visualizzazione
        print("\n" + "─" * 60)
        print("📝 RIEPILOGO ANNOTAZIONE:")
        print("─" * 60)
        print(f"   🎥 Angolo:        {angolo_camera}")
        print(f"   👤 Visibilità:    {visibilita}")
        print(f"   🦶 Piede:         {piede}")
        print(f"   ⚡ Velocità:      {velocita}")
        print(f"   🎭 Fake:          {fake}")
        print(f"   📍 Posizione:     {lato} - {altezza}")
        print(f"   ⚽ Dentro/Fuori:  {dentro_fuori}")
        print(f"   🧤 Parato:        {parato}")
        print("─" * 60)

        # Warning se angolo non ideale
        if angolo_camera != 'lateral':
            print("⚠️  ATTENZIONE: Angolo non laterale - pose estimation potrebbe fallire")
        if visibilita == 'poor':
            print("⚠️  ATTENZIONE: Visibilità scarsa - considera di escludere questo video")

        confirm = input("\n✓ Confermi? [y/n/s=skip] → ").strip().lower()

        if confirm == 'y':
            return annotation
        elif confirm == 's':
            return None
        else:
            print("\n↻ Riprovo...\n")
            return self.annotate_video(video_file)

    def save_annotation(self, annotation: Dict):
        """Salva singola annotazione"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'video_file', 'piede', 'lato', 'altezza', 'dentro_fuori', 'parato',
                'angolo_camera', 'visibilita_giocatore', 'velocita_rincorsa', 'fake'
            ])
            writer.writerow(annotation)
        self.annotations.append(annotation)

    def run(self, batch_size: int = 50):
        """Esegue annotazione batch"""
        # Ottieni tutti i video
        videos = sorted(self.video_dir.glob("*.mp4"))

        if not videos:
            print("❌ Nessun video trovato in", self.video_dir)
            return

        # Filtra già annotati
        annotated = self._get_annotated_files()
        to_annotate = [v for v in videos if v.name not in annotated]

        # Limita a batch_size
        to_annotate = to_annotate[:batch_size]

        if not to_annotate:
            print("✓ Tutti i video sono già annotati!")
            self._print_stats()
            return

        print(f"\n🎬 Trovati {len(videos)} video totali")
        print(f"✓ Già annotati: {len(annotated)}")
        print(f"📝 Da annotare: {len(to_annotate)}")
        print(f"\n{'=' * 60}")

        input("Premi INVIO per iniziare...")

        # Annota
        skipped = 0
        for i, video_path in enumerate(to_annotate, 1):
            print(f"\n\n{'#' * 60}")
            print(f"# [{i}/{len(to_annotate)}] - Completati: {i - 1 - skipped}")
            print(f"{'#' * 60}")

            # Apri video
            self._play_video(video_path)
            print("\n⏸  Video aperto. Guardalo attentamente (anche 2-3 volte).")
            input("Premi INVIO quando pronto ad annotare...")

            # Annota
            annotation = self.annotate_video(video_path.name)

            if annotation:
                self.save_annotation(annotation)
                print(f"\n✅ Salvato! (Totale: {len(self.annotations)})")
            else:
                skipped += 1
                print(f"\n⊘ Saltato (Saltati: {skipped})")

            # Statistiche progressive
            if i % 10 == 0:
                self._print_stats()

        # Fine
        print(f"\n\n{'=' * 60}")
        print("🎉 ANNOTAZIONE COMPLETATA!")
        print('=' * 60)
        self._print_stats()

    def _print_stats(self):
        """Stampa statistiche dettagliate annotazioni"""
        if not self.annotations:
            return

        from collections import Counter

        # Conta
        piedi = Counter(a['piede'] for a in self.annotations)
        lati = Counter(a['lato'] for a in self.annotations)
        altezze = Counter(a['altezza'] for a in self.annotations)
        dentro_fuori = Counter(a['dentro_fuori'] for a in self.annotations)
        parati = Counter(a['parato'] for a in self.annotations if a['parato'] != 'n/a')

        # Combinazioni lato+altezza (griglia 3x3)
        posizioni = Counter(f"{a['lato']}-{a['altezza']}" for a in self.annotations)

        print(f"\n{'=' * 60}")
        print(f"📊 STATISTICHE ({len(self.annotations)} rigori annotati)")
        print('=' * 60)

        angoli = Counter(a['angolo_camera'] for a in self.annotations)
        visibilita = Counter(a['visibilita_giocatore'] for a in self.annotations)
        velocita = Counter(a['velocita_rincorsa'] for a in self.annotations)
        fake = Counter(a['fake'] for a in self.annotations)

        print(f"\n🎥 Angolo Camera:")
        for ang, count in angoli.most_common():
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            warn = " ⚠️" if ang != 'lateral' else ""
            print(f"   {ang:15s}: {count:3d} ({pct:5.1f}%) {bar}{warn}")

        print(f"\n👤 Visibilità:")
        for vis, count in visibilita.most_common():
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            warn = " ⚠️" if vis == 'poor' else ""
            print(f"   {vis:15s}: {count:3d} ({pct:5.1f}%) {bar}{warn}")

        print(f"\n⚡ Velocità Rincorsa:")
        for vel, count in velocita.most_common():
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            print(f"   {vel:15s}: {count:3d} ({pct:5.1f}%) {bar}")

        print(f"\n🎭 Fake/Deception:")
        for f, count in fake.most_common():
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            print(f"   {f:15s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Piede
        print(f"\n🦶 Piede:")
        for p, count in piedi.most_common():
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            print(f"   {p:10s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Lato
        print(f"\n📍 Lato (orizzontale):")
        for l, count in sorted(lati.items()):
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            print(f"   {l:10s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Altezza
        print(f"\n📏 Altezza (verticale):")
        for a, count in sorted(altezze.items()):
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            print(f"   {a:10s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Griglia 3x3
        print(f"\n🎯 Heatmap Posizioni (griglia 3x3):")
        grid_order = [
            'left-high', 'center-high', 'right-high',
            'left-mid', 'center-mid', 'right-mid',
            'left-low', 'center-low', 'right-low'
        ]
        for i, pos in enumerate(grid_order):
            count = posizioni[pos]
            if i % 3 == 0:
                print()
            pct = count / len(self.annotations) * 100 if count > 0 else 0
            print(f"   [{count:2d}] {pct:4.1f}%", end="  ")
        print()

        # Dentro/Fuori
        print(f"\n⚽ Esito:")
        for df, count in dentro_fuori.most_common():
            pct = count / len(self.annotations) * 100
            bar = '█' * int(pct / 2)
            print(f"   {df:10s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Parati (solo tra quelli dentro)
        if parati:
            dentro_count = dentro_fuori['dentro']
            print(f"\n🧤 Parati (su {dentro_count} dentro):")
            for p, count in parati.most_common():
                pct = count / dentro_count * 100
                bar = '█' * int(pct / 2)
                print(f"   {p:10s}: {count:3d} ({pct:5.1f}%) {bar}")

        # Bilanciamento
        print(f"\n⚖️  Bilanciamento:")
        if 'left' in lati and 'right' in lati:
            ratio = lati['left'] / lati['right']
            balanced = 0.7 < ratio < 1.3
            print(f"   Left/Right ratio: {ratio:.2f} {'✓' if balanced else '⚠️'}")

        print('=' * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool annotazione dettagliata rigori"
    )
    parser.add_argument(
        '--video-dir',
        default=r'C:\Users\sprochilo\PycharmProjects\PenaltyVision\penalty_clips',
        help='Directory con video clips'
    )
    parser.add_argument(
        '--csv-file',
        default=r'C:\Users\sprochilo\PycharmProjects\PenaltyVision\annotations.csv',
        help='File CSV output annotazioni'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=50,
        help='Numero video da annotare in questa sessione'
    )

    args = parser.parse_args()

    # Istruzioni
    print("""
╔══════════════════════════════════════════════════════════╗
║     TOOL ANNOTAZIONE DETTAGLIATA RIGORI                  ║
╚══════════════════════════════════════════════════════════╝

📝 COSA ANNOTARE (5 attributi):

   1. 🦶 PIEDE calciante (destro/sinistro)
   
   2. 📍 POSIZIONE sulla porta (griglia 3x3):
      - Lato: sinistra / centro / destra
      - Altezza: alto / medio / basso
   
   3. ⚽ DENTRO o FUORI porta
      - Dentro: palla entra in porta (anche se parata)
      - Fuori: palo, traversa, o completamente fuori
   
   4. 🧤 PARATO o GOL (solo se dentro)
      - Parato: portiere tocca/blocca
      - Gol: entra in rete

💡 TIPS:
   - Guarda il video 2-3 volte, anche in slow-mo
   - La griglia è dalla prospettiva telecamera
   - Se non sei sicuro, puoi saltare (s) e fare dopo
   - Interrompi quando vuoi (Ctrl+C), riprendi dopo
   
⌨️  SHORTCUT:
   - r/l = right/left
   - d/f = dentro/fuori  
   - p/g = parato/gol
   - 1-9 = posizione griglia
   - y/n/s = yes/no/skip

""")

    input("Premi INVIO per iniziare...")

    # Avvia annotazione
    annotator = PenaltyAnnotator(
        video_dir=args.video_dir,
        csv_file=args.csv_file
    )

    try:
        annotator.run(batch_size=args.batch)
    except KeyboardInterrupt:
        print("\n\n⏸  Annotazione interrotta")
        print(f"✓ Salvate {len(annotator.annotations)} annotazioni")
        print(f"📁 File: {annotator.csv_file}")
        annotator._print_stats()


if __name__ == "__main__":
    main()
