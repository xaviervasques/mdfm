"""The input contract of the generator, Wan2.2-Animate-14B in animation mode, read from its source.

Read, never imported: DiffSynth-Studio is installed in the mdfm_video environment only (env/VIDEO_ENV.md), at
<store>/src/DiffSynth-Studio, commit DIFFSYNTH_COMMIT. File and line references, also in
reports/video_driver/summary.md:

- diffsynth/pipelines/wan_video.py: WanVideoPipeline.__call__ takes the reference image (input_image, a PIL
  image), the pose video and the face video (animate_pose_video, animate_face_video: lists of PIL images),
  height, width, num_frames (lines 229-251; defaults 480, 832, 81). The shape check rounds height and width up
  to multiples of 16 and num_frames up to 4 k + 1 (line 38; diffsynth/diffusion/base_pipeline.py lines
  98-116). The reference image is resized to (width, height) without keeping its aspect (lines 491, 514), so it
  is generated at the pose video's size. The pose video is VAE-encoded as it is, never resized (lines 980-994;
  base_pipeline.py lines 118-131), so its frames are width x height; its latents are added to every latent
  frame after the first (diffsynth/models/wan_video_animate_adapter.py line 625), so a call of num_frames
  frames takes num_frames - 4 pose frames (one reference latent plus (num_frames - 4 - 1) / 4 + 1 pose
  latents), and the reference latent is dropped before decoding (wan_video.py lines 357-362): 81 frames per
  call give 77 output frames. The face frames are 512 x 512 (the motion encoder is built for size 512,
  wan_video_animate_adapter.py line 619; its last convolution leaves a 4 x 4 map only from 512 x 512), as many
  as the pose frames (two stride-2 causal convolutions map 77 face frames to the 20 pose latents, lines 73-77).
- examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-14B.py: 77 pose and face frames per call
  (lines 40-41), num_frames 81, height 720, width 1280 (line 48), 20 steps, cfg_scale 1 (line 49), seed 0
  (line 44), disk and CPU offload under a VRAM limit of the card's memory minus 2 GiB (lines 9-29). The example
  saves its output with fps=15 (line 51): a container rate only, the pipeline takes no frame rate.
- The pipeline generates one segment per call; it has no argument named for guidance frames for this model
  (the refert frames of wan_video.py line 1255 belong to Wan-Animate-2), but its replace-mode inputs carry
  them (task VCH, reading of 2026-10-05): WanVideoUnit_AnimateInpaint (lines 1013-1054) builds the DiT's
  conditioning y from animate_inpaint_video and animate_mask_video as Wan2.2's continuation builds it (the
  previous segment's last frames with mask 1, zeros with mask 0). A clip longer than 77 frames is a sequence of
  calls chained by that scheme (scripts/video_generate.py --chain wan, which cites the lines).

From the Wan2.2 repository (github.com/Wan-Video/Wan2.2, branch main, read on 2026-10-04 and 2026-10-05; the
V0 fetch output in reports/weekend_2026-10-01/tasks/V0.md, [F3] and [R1]): sample_fps 30, frame_num 77 and
vae_stride (4, 8, 8) in wan/configs/wan_animate_14B.py; preprocess_data.py resamples the driving video to
--fps 30 and resizes frames by area to 1280 x 720 with sides divisible by 16; process_pipepline.py draws the
pose on a black canvas of the frame's size with draw_aapose_by_meta_new (threshold 0.5, stickwidth v2, hands
drawn) and crops the face to 512 x 512; wan/animate.py advances a segment by clip_len - refert_num frames and
pads the input to whole segments. The drawing itself is mdfm/video/draw.py.

The generator has no frame-rate input: a pose video drawn at 60 fps is read as frames. The 60 fps arm of the
proposal (section 3.8) is a pose video sampled at 60 fps whose output is decimated to 30 fps before estimation.
"""
import math

DIFFSYNTH_COMMIT = "974cfa37f27ac55eba3b6d10efa21f876900572d"
SAMPLE_FPS = 30             # Wan2.2 wan_animate_14B.py sample_fps; preprocess_data.py --fps default
SEGMENT_FRAMES = 77         # Wan2.2 frame_num; DiffSynth: num_frames - 4 pose frames per call
CALL_FRAMES = 81            # DiffSynth num_frames of one call (the example, line 48)
REFERT_FRAMES = 1           # guidance frames between segments (proposal 3.1; wan/animate.py takes 1 or 5)
TIME_STRIDE = 4             # Wan VAE temporal stride (vae_stride (4, 8, 8); DiffSynth time_division_factor 4)
SPATIAL_DIVISOR = 16        # DiffSynth height and width division factor
FACE_SIZE = 512             # face crop side, px
DEFAULT_SIZE = (1280, 720)  # width, height: Wan2.2 --resolution_area default and the DiffSynth example
LOW_SIZE = (832, 480)       # width, height: the DiffSynth pipeline defaults (a 480P-class size)


def check_size(width, height):
    """(width, height) unchanged when both are positive multiples of 16 (the shape check would otherwise
    round them up and the pose latents would no longer match); ValueError otherwise."""
    w, h = int(width), int(height)
    if w != width or h != height or w <= 0 or h <= 0 or w % SPATIAL_DIVISOR or h % SPATIAL_DIVISOR:
        raise ValueError(f"width and height must be positive multiples of {SPATIAL_DIVISOR}, got {width} x {height}")
    return w, h


def latents_per_second(fps):
    """Pose latents per second of a pose video at fps (temporal stride 4): 7.5 at 30 fps, 15 at 60 fps."""
    return float(fps) / TIME_STRIDE


def latent_count(n_frames):
    """Latent frames of n_frames pose frames: (n - 1) / 4 + 1, n = 4 k + 1 (77 frames: 20 latents)."""
    n = int(n_frames)
    if n < 1 or (n - 1) % TIME_STRIDE:
        raise ValueError(f"a pose video of {n_frames} frames is not 4 k + 1 frames")
    return (n - 1) // TIME_STRIDE + 1


def call_frames(n_pose_frames=SEGMENT_FRAMES):
    """num_frames of the DiffSynth call that takes n_pose_frames pose and face frames: n + 4 (77 -> 81)."""
    latent_count(n_pose_frames)
    return int(n_pose_frames) + 4


def segment_plan(n_frames, segment=SEGMENT_FRAMES, refert=REFERT_FRAMES):
    """Segments of a clip of n_frames pose frames, as wan/animate.py chains them: segment i covers frames
    [s_i, s_i + segment), s_i = i (segment - refert); its first refert frames are guidance frames copied from
    the previous segment's output, so its own frames start at s_i + refert (a seam, for i >= 1). The last
    segment is padded to whole length. Returns {segment, refert, stride, n_frames, n_segments, starts, seams,
    pad}."""
    n, seg, r = int(n_frames), int(segment), int(refert)
    if n < 1 or seg < 1 or not 0 <= r < seg:
        raise ValueError(f"bad segment plan: {n_frames} frames, segment {segment}, refert {refert}")
    stride = seg - r
    k = 1 + max(0, math.ceil((n - seg) / stride))
    starts = [i * stride for i in range(k)]
    return {"segment": seg, "refert": r, "stride": stride, "n_frames": n, "n_segments": k, "starts": starts,
            "seams": [s + r for s in starts[1:]], "pad": starts[-1] + seg - n}


def contract(fps=SAMPLE_FPS, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1]):
    """The contract as a dict, for the provenance record of a clip."""
    check_size(width, height)
    return {"generator": "Wan2.2-Animate-14B, animation mode, through DiffSynth-Studio",
            "diffsynth_commit": DIFFSYNTH_COMMIT, "pose_video": f"{width} x {height} RGB frames, black canvas",
            "face_video": f"{FACE_SIZE} x {FACE_SIZE} RGB frames, as many as the pose frames",
            "reference_image": f"{width} x {height} (resized without keeping the aspect otherwise)",
            "frames_per_call": CALL_FRAMES, "pose_frames_per_call": SEGMENT_FRAMES,
            "refert_frames": REFERT_FRAMES, "time_stride": TIME_STRIDE, "sample_fps": SAMPLE_FPS,
            "driver_fps": float(fps), "latents_per_second": latents_per_second(fps)}
