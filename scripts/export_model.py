import argparse
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the trained shot classifier to ONNX for Android."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("shot_classifier.pkl"),
        help="Path to the trained sklearn classifier.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("shot_classifier.onnx"),
        help="Where to save the ONNX model.",
    )
    parser.add_argument(
        "--feature-count",
        type=int,
        default=5,
        help="Number of float features expected by the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = joblib.load(args.model_path)
    initial_type = [("float_input", FloatTensorType([None, args.feature_count]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    args.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    args.onnx_path.write_bytes(onnx_model.SerializeToString())

    print(f"Model exported to: {args.onnx_path.resolve()}")


if __name__ == "__main__":
    main()
