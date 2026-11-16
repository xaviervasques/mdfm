"""Pose-driven synthetic video, the CPU side (docs/proposals/VIDEO_SYNTH_ROUNDTRIP_DRAFT.md, section 7, step 2).

- contract: the input contract of the generator (Wan2.2-Animate-14B through DiffSynth-Studio), read from its
  source: sizes, frames per call and per segment, face crops, frame rate, latent stride, segment plan.
- points: the drawn point set of the Wan2.2 pose drawing (20 body points, 21 per hand) from an SMPL+H body.
- draw: that drawing (draw_aapose_by_meta_new), re-implemented in numpy.
- render: a body clip rendered into a driver pose video at 30 or 60 fps, with its framing, truth and provenance.
- labels: the SYNTHETIC label rules of section 6 and the output root <store>/video.
- analysis: the measures of section 3.4 and their void check.
- exam_driver (task VCH): the synthetic patient's bodies (mdfm/synth) as BodyClips, with their operator's truth,
  and the exam driver set exam_v1 (scripts/render_driver_video.py --exam v1).
- seams (task VCH): the keypoint measures of the seams of a chained generated clip (phase jumps, spurious jerk
  onsets), for the stage-1 round trip.

Nothing here imports DiffSynth or uses the GPU, and no existing behaviour of the program changes: the package is
new and only scripts/render_driver_video.py and scripts/video_roundtrip_analysis.py call it.
"""
