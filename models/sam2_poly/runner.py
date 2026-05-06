from __future__ import annotations

from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("sam2_seg")
    output_dir = ctx.output_dir()
    pred_path = output_dir / "predictions.json"
    bbox_value = ctx.param("MASKRCNN_BBOX_JSON", "")
    bbox_json = ctx.bbox_json or (Path(bbox_value) if bbox_value else ctx.output_root / "maskrcnn_poly" / ctx.dataset / ctx.task / f"maskrcnn_poly_{ctx.task}_train_eval" / "bbox_predictions_for_sam2.json")
    if not bbox_json.is_absolute():
        bbox_json = ctx.release_root / bbox_json
    require_paths([source / "run_sam2_inference.py", data_root / "val"])

    cwd = ctx.release_root
    steps = [
        CommandStep(
            "sam2_poly_infer",
            cwd=cwd,
            env=ctx.env(),
            argv=ctx.python(env_name, source / "run_sam2_inference.py", [
                "--mirror-root", ctx.path(data_root, cwd),
                "--bbox-json", ctx.path(bbox_json, cwd),
                "--output", ctx.path(pred_path, cwd),
                "--score-threshold", ctx.param("SCORE_THRESHOLD", 0.5),
                "--connectivity", ctx.param("CONNECTIVITY", 4),
                "--simplify-tol", ctx.param("SIMPLIFY_TOL", 1.0),
                "--min-area", ctx.param("MIN_AREA", 16),
                "--min-hole-area", ctx.param("MIN_HOLE_AREA", 16),
                "--sam2-model-id", ctx.param("SAM2_MODEL_ID", "facebook/sam2-hiera-small"),
                "--max-images", ctx.param("MAX_INFER_IMAGES", 4 if ctx.smoke else 0),
            ], cwd),
        )
    ]
    if ctx.run_val:
        steps.append(ctx.evaluate_step(name="sam2_poly_eval", pred=pred_path, env_name=env_name, cwd=cwd))
    return steps
