#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run camera-only teacher pipeline end-to-end for one or more bags."
    )
    parser.add_argument("--bag", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--sample-every-n", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--flow-image-side-px", type=int, default=320)
    parser.add_argument("--flow-motion-threshold", type=float, default=1.5)

    parser.add_argument("--model", default="qwen2.5vl:32b-q4_K_M")
    parser.add_argument(
        "--prompt-mode",
        choices=["metadata", "class_only", "camera_reason_temporal"],
        default="camera_reason_temporal",
    )
    parser.add_argument("--annotate-limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument("--max-image-side-px", type=int, default=256)
    parser.add_argument("--annotate-jpeg-quality", type=int, default=60)
    parser.add_argument("--num-predict", type=int, default=96)
    parser.add_argument("--num-ctx", type=int, default=384)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--prepare-min-class-count", type=int, default=1)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-annotate", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def run_step(cmd, cwd):
    print("\n[run] {}".format(" ".join(str(part) for part in cmd)), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    bag_paths = [Path(bag).expanduser().resolve() for bag in args.bag]

    python_bin = sys.executable

    if not args.skip_export:
        cmd = [
            python_bin,
            str(project_dir / "scripts" / "export_camera_only_teacher_dataset.py"),
            "--bag",
            *[str(path) for path in bag_paths],
            "--output-dir",
            str(output_dir),
            "--image-topic",
            str(args.image_topic),
            "--sample-every-n",
            str(int(args.sample_every_n)),
            "--jpeg-quality",
            str(int(args.jpeg_quality)),
            "--flow-image-side-px",
            str(int(args.flow_image_side_px)),
            "--flow-motion-threshold",
            str(float(args.flow_motion_threshold)),
        ]
        if int(args.max_samples) > 0:
            cmd.extend(["--max-samples", str(int(args.max_samples))])
        run_step(cmd, project_dir)

    if not args.skip_annotate:
        cmd = [
            python_bin,
            str(project_dir / "scripts" / "annotate_teacher_with_ollama.py"),
            "--dataset-dir",
            str(output_dir),
            "--model",
            str(args.model),
            "--prompt-mode",
            str(args.prompt_mode),
            "--timeout-s",
            str(float(args.timeout_s)),
            "--retries",
            str(int(args.retries)),
            "--retry-sleep-s",
            str(float(args.retry_sleep_s)),
            "--max-image-side-px",
            str(int(args.max_image_side_px)),
            "--jpeg-quality",
            str(int(args.annotate_jpeg_quality)),
            "--num-predict",
            str(int(args.num_predict)),
            "--num-ctx",
            str(int(args.num_ctx)),
            "--keep-alive",
            str(args.keep_alive),
        ]
        if int(args.annotate_limit) > 0:
            cmd.extend(["--limit", str(int(args.annotate_limit))])
        if args.prewarm:
            cmd.append("--prewarm")
        if args.overwrite:
            cmd.append("--overwrite")
        run_step(cmd, project_dir)

    if not args.skip_prepare:
        cmd = [
            python_bin,
            str(project_dir / "scripts" / "prepare_teacher_labels.py"),
            "--dataset-dir",
            str(output_dir),
            "--min-class-count",
            str(int(args.prepare_min_class_count)),
        ]
        run_step(cmd, project_dir)

    if not args.skip_train:
        cmd = [
            python_bin,
            str(project_dir / "scripts" / "train_student_baseline.py"),
            "--dataset-dir",
            str(output_dir),
        ]
        run_step(cmd, project_dir)

    print("\n[pipeline] done")
    print("[pipeline] output_dir={}".format(output_dir))


if __name__ == "__main__":
    main()
