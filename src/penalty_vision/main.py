import argparse
from pathlib import Path

from penalty_vision.processor.penalty_kick_feature_extractor import PenaltyKickFeatureExtractor
from penalty_vision.utils import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_dir = Path(config.paths.output) / 'embedding'

    with PenaltyKickFeatureExtractor(args.config, str(output_dir)) as extractor:
        results = extractor.process_dataset_from_csv(args.annotations, config.paths.video_dir)

    print(f"\nProcessed: {len(results['successful'])}/{len(results['successful']) + len(results['failed'])}")
    if results['failed']:
        print(f"Failed: {', '.join(results['failed'])}")
