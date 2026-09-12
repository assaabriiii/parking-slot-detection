from parking_mvp.io_schema import SpotStatus
from parking_mvp.temporal import TemporalSmoother


def _spot(spot_id: str, status: str) -> SpotStatus:
    return SpotStatus(spot_id=spot_id, status=status, confidence=0.8, head="opencv")


def test_majority_occupied():
    sm = TemporalSmoother(window=5, min_votes=3)
    last = None
    for label in ["occupied", "occupied", "free", "occupied", "occupied"]:
        last = sm.update([_spot("s1", label)])[0]
    assert last is not None
    assert last.status == "occupied"
    assert last.raw_status == "occupied"


def test_tie_keeps_previous():
    sm = TemporalSmoother(window=2, min_votes=1)
    sm.update([_spot("s1", "free")])
    out = sm.update([_spot("s1", "occupied")])[0]
    assert out.status == "free"


def test_independent_spots():
    sm = TemporalSmoother(window=3, min_votes=2)
    sm.update([_spot("a", "free"), _spot("b", "occupied")])
    sm.update([_spot("a", "free"), _spot("b", "occupied")])
    out = {s.spot_id: s.status for s in sm.update([_spot("a", "occupied"), _spot("b", "free")])}
    assert out["a"] == "free"
    assert out["b"] == "occupied"
