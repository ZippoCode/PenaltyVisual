import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional


class PenaltyAnnotator:
    def __init__(self, video_dir: str, csv_file: str):
        self.video_dir = Path(video_dir)
        self.csv_file = Path(csv_file)
        self.annotations = []
        self.fieldnames = [
            'video_file', 'foot', 'side', 'height', 'in_out', 'saved',
            'camera_angle', 'player_visibility', 'run_speed', 'fake'
        ]
        self._load_existing()

    def _load_existing(self):
        """Load existing annotations or create new CSV"""
        if self.csv_file.exists():
            with open(self.csv_file, 'r') as f:
                self.annotations = list(csv.DictReader(f))
            print(f"Loaded {len(self.annotations)} annotations")
        else:
            self.csv_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_file, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
            print("Created new annotations.csv")

    def _get_annotated_files(self) -> set:
        """Get set of already annotated files"""
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

    def _get_input(self, prompt: str, valid: List[str]) -> str:
        """Get validated user input"""
        while True:
            response = input(prompt).strip().lower()
            if response in valid:
                return response
            print(f"Invalid. Choose: {', '.join(valid)}")

    def annotate_video(self, video_file: str) -> Optional[Dict]:
        """Annotate single video"""
        print(f"\n{'=' * 60}\nVideo: {video_file}\n{'=' * 60}")

        # Camera angle
        print("\nCAMERA ANGLE: [l]ateral [d]iagonal [b]ehind [f]rontal")
        angle = self._get_input("→ ", ['l', 'd', 'b', 'f'])
        angle_map = {'l': 'lateral', 'd': 'diagonal', 'b': 'behind', 'f': 'frontal'}

        # Player visibility
        print("\nPLAYER VISIBILITY: [f]ull [p]artial [o]bstructed")
        vis = self._get_input("→ ", ['f', 'p', 'o'])
        vis_map = {'f': 'full', 'p': 'partial', 'o': 'obstructed'}

        # Kicking foot
        foot = self._get_input("\nFOOT: [r]ight [l]eft → ", ['r', 'l'])
        foot = "right" if foot == 'r' else "left"

        # Run-up speed
        print("\nRUN SPEED: [s]low [m]edium [f]ast")
        speed = self._get_input("→ ", ['s', 'm', 'f'])
        speed_map = {'s': 'slow', 'm': 'medium', 'f': 'fast'}

        # Fake/deception
        fake = self._get_input("\nFAKE: [y]es [n]o → ", ['y', 'n'])
        fake = "yes" if fake == 'y' else "no"

        # In/out
        in_out = self._get_input("\nRESULT: [i]n [o]ut → ", ['i', 'o'])
        in_out = "in" if in_out == 'i' else "out"

        # Saved (only if in)
        saved = "n/a"
        if in_out == "in":
            s = self._get_input("\nSAVED: [s]aved [g]oal → ", ['s', 'g'])
            saved = "saved" if s == 's' else "goal"

        # Position grid (3x3)
        print("\nPOSITION GRID:\n  1-L.High  2-C.High  3-R.High")
        print("  4-L.Mid   5-C.Mid   6-R.Mid")
        print("  7-L.Low   8-C.Low   9-R.Low")
        pos = self._get_input("\nPosition [1-9] → ", [str(i) for i in range(1, 10)])

        grid = {
            '1': ('left', 'high'), '2': ('center', 'high'), '3': ('right', 'high'),
            '4': ('left', 'mid'), '5': ('center', 'mid'), '6': ('right', 'mid'),
            '7': ('left', 'low'), '8': ('center', 'low'), '9': ('right', 'low')
        }
        side, height = grid[pos]

        annotation = {
            'video_file': video_file,
            'foot': foot,
            'side': side,
            'height': height,
            'in_out': in_out,
            'saved': saved,
            'camera_angle': angle_map[angle],
            'player_visibility': vis_map[vis],
            'run_speed': speed_map[speed],
            'fake': fake
        }

        # Summary
        print(f"\nSUMMARY: {angle_map[angle]} | {vis_map[vis]} | {foot} | {speed_map[speed]}")
        print(f"         {fake} fake | {side}-{height} | {in_out} | {saved}")

        confirm = input("\nConfirm? [y/n/s=skip] → ").strip().lower()

        if confirm == 'y':
            return annotation
        elif confirm == 's':
            return None
        else:
            return self.annotate_video(video_file)

    def save_annotation(self, annotation: Dict):
        """Save single annotation"""
        with open(self.csv_file, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(annotation)
        self.annotations.append(annotation)

    def run(self, batch_size: int = 50):
        """Run batch annotation"""
        videos = sorted(self.video_dir.glob("*.mp4"))
        if not videos:
            print(f"No videos found in {self.video_dir}")
            return

        annotated = self._get_annotated_files()
        to_annotate = [v for v in videos if v.name not in annotated][:batch_size]

        if not to_annotate:
            print("All videos annotated!")
            self._print_stats()
            return

        print(f"\nTotal: {len(videos)} | Annotated: {len(annotated)} | To do: {len(to_annotate)}")
        input("Press ENTER to start...")

        skipped = 0
        for i, video_path in enumerate(to_annotate, 1):
            print(f"\n[{i}/{len(to_annotate)}] Completed: {i - 1 - skipped}")

            self._play_video(video_path)
            input("Video opened. Press ENTER when ready...")

            annotation = self.annotate_video(video_path.name)

            if annotation:
                self.save_annotation(annotation)
                print(f"Saved! Total: {len(self.annotations)}")
            else:
                skipped += 1
                print(f"Skipped (total skipped: {skipped})")

        print(f"\n{'=' * 60}\nANNOTATION COMPLETE!\n{'=' * 60}")
        self._print_stats()

    def _print_stats(self):
        """Print annotation statistics"""
        if not self.annotations:
            return

        n = len(self.annotations)
        print(f"\nSTATISTICS ({n} penalties)\n{'=' * 60}")

        for field in ['camera_angle', 'player_visibility', 'foot', 'run_speed', 'fake', 'side', 'height', 'in_out']:
            counts = Counter(a[field] for a in self.annotations)
            print(f"\n{field.upper()}:")
            for val, count in counts.most_common():
                pct = count / n * 100
                print(f"  {val:15s}: {count:3d} ({pct:5.1f}%)")

        # Grid heatmap
        print("\nGRID HEATMAP:")
        positions = Counter(f"{a['side']}-{a['height']}" for a in self.annotations)
        grid_order = [
            'left-high', 'center-high', 'right-high',
            'left-mid', 'center-mid', 'right-mid',
            'left-low', 'center-low', 'right-low'
        ]
        for i, pos in enumerate(grid_order):
            if i % 3 == 0:
                print()
            count = positions[pos]
            pct = count / n * 100 if count else 0
            print(f"  [{count:2d}] {pct:4.1f}%", end="  ")
        print("\n" + "=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Penalty annotation tool")
    parser.add_argument('--video-dir', required=True, help='Video clips directory')
    parser.add_argument('--csv-file', required=True, help='Output CSV file')
    parser.add_argument('--batch', type=int, default=50, help='Batch size')
    args = parser.parse_args()

    print("\nPENALTY ANNOTATION TOOL")
    print("Annotate: camera angle, visibility, foot, speed, fake, position, result\n")
    input("Press ENTER to start...")

    annotator = PenaltyAnnotator(args.video_dir, args.csv_file)

    try:
        annotator.run(batch_size=args.batch)
    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Saved {len(annotator.annotations)} annotations")
        annotator._print_stats()


if __name__ == "__main__":
    main()
