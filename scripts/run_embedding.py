import argparse
from pathlib import Path

from penalty_vision.processing import HAREmbeddingExtractor, PenaltyKickPipeline
from penalty_vision.utils import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=False, default='embeddings')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_dir = Path(config.paths.output) / args.output_dir
    har_extractor = HAREmbeddingExtractor(device="mps")
    with PenaltyKickPipeline(args.config, har_extractor, str(output_dir)) as extractor:
        results = extractor.process_dataset_from_csv(args.annotations, config.paths.video_dir)

    print(f"\nProcessed: {len(results['successful'])}/{len(results['successful']) + len(results['failed'])}")
    if results['failed']:
        print(f"Failed: {', '.join(results['failed'])}")
