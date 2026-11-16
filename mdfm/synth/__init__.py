"""The synthetic patient's body side (step SP1 of
docs/proposals/EXPLORATION_PLAN_SYNTHETIC_PATIENT_AND_JEPA_2026-10-05.md; exam and grade maps of
docs/proposals/SYNTH_PATIENT_DRAFT.md, decisions D3 and D4 as proposed, recorded in docs/OWNER_ACTIONS.md
section 2).

Pathology operators that act on one body trajectory in the program's body representation (rot6d of the 22
joints with the root orientation in row 0, the translation, foot contacts, the 100 Hz clock) and return
the modified trajectory with its ground truth in physical units:
- tremor.tremor: rest and postural forearm tremor (Hz, peak-to-peak wrist displacement in cm);
- freezing.freezing, freezing.plan_episodes, freezing.freezing_grade: freezing of gait episodes (onset,
  duration, akinesia or trembling in the 3 to 8 Hz freeze band, stance feet locked);
- bradykinesia.bradykinesia: repeated tasks of MDS-UPDRS 3.6 to 3.8 with decrement, hesitations and arrests;
- dystonia.dystonic_posture: sustained postures from the TWSTRS and DIS grade maps, bounded by the SP1
  joint-limit table.
Each operator maps a grade to parameters by grades.py, is deterministic under its seed, returns a copy of its
input bit for bit at grade 0, and reports the kinematic checks of body.validity (joint limits, foot skate in
stance, contacts) in its truth record (common.truth_record).

Support: body.py (the Body container, forward kinematics, the joint-limit table, the checks), base.py
(procedural postures and a footstep walker), ik.py (leg inverse kinematics), contact.py (the foot on the floor:
sole cloud, rolling foot, planted legs; task SP4), measure.py (read-back of frequencies, amplitudes, cycles and
deviations), stimuli.py (the stimulus set of scripts/sp3_decoder_prior_probe.py and the grade levels in physical
units).

Viewer (task SP4, version (a) of draft section 9): raster.py (a numpy z-buffer rasterizer, flat or Gouraud
shading), viewer.py (the mesh of a body at 30 frames per second from clinical views, the SYNTHETIC mark and
caption, MP4 and PNG written only under <store>/synth_viewer), exam.py (the first exam set: the healthy references
by the program's generators since exam_v10, the four families at the draft's grades);
scripts/render_mesh_video.py renders it with an index page.

Exam motions (task SP5, exam_v3): hands.py (the 30 hand joints: HandBody, skinning with them,
anatomical axes, fingertips, the cited hand joint limits), handtasks.py (finger tapping 3.4 and hand movements 3.5
at cited normal rates and amplitudes, with decrement, interruptions, arrest and the slowing ladder anchored to the
cited normal rate; bradykinesia.bradykinesia dispatches to it), motions.py (arm postures, the raise to the 3.15
posture with the fingers apart, the arising of 3.9 from a cited seat height at the cited duration and phases).

Body model (plan phase C, docs/OPEN_BODY_PORT_DESIGN.md): openbody.py (the open body, Anny, the default since wave
1) and smplh_reference.py (the SMPL+H body, reached only behind its flag for the port's comparison), selected by
body.body_model. Since wave 8 no module here imports mdfm.body.smplh, the SMPL+H reference or the AMASS modules at
its top: the joint table and fk come from mdfm/body/joints.py, and the SMPL+H and AMASS code of the synthetic patient
(the reference body, the SkeletonBank rest, the exam set's former AMASS references) lives in smplh_reference.py,
imported inside its reference branches only; that module never ships (NEVER_SHIP of scripts/build_public_tree.py).

Use: test and stimulus instrument only. Synthetic pathology never feeds the deviation embedding, a calibrator,
a readout, a gate threshold or any training mixture (plan principle 3; design section 5). Nothing here reads a
lockbox, EVAL, CODY or care_pd record.
"""
