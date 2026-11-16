"""The AMASS take the body, kernel and factory tests share (B4.1).

KIT/674/dry_head01_poses.npz: the first member of the KIT archive (a train archive, so a cache rebuild
reads only the head of the bz2 stream), a standing take of 1,109 frames at 100 Hz (manifest), female,
whose MoveAtlas canonical record is rigid (KIT has no corrupted canonical record,
reports/probe_amass_canonical_rigidity/summary.md).
"""
TAKE = ("KIT", "KIT/674/dry_head01_poses.npz")
SUBJECT_NS = "amass:KIT:KIT_674"


def canonical_row(take=TAKE, subject_ns=SUBJECT_NS):
    """The join keys of the take's canonical record (mdfm.data.canonical.load)."""
    return {"source": "amass", "modality": "pose", "subject_ns": subject_ns, "key": take[1]}


def assert_train_take(take=TAKE, subject_ns=SUBJECT_NS):
    """The take sits in a train archive, belongs to a person outside the lockbox and passes the loader's
    guard without any flag."""
    from mdfm.data import amass_index as I
    archive, member = take
    assert I.archive_split(archive) == "train"
    assert I.archive_of_member(member) == archive
    rows = [r for r in I.FOLDERS if r["subject_ns"] == subject_ns]
    assert len(rows) == 1 and rows[0]["archive"] == archive
    assert rows[0]["person_id"] not in I.LOCKBOX_PERSONS
    assert I.check_member(archive, member) == "train"
