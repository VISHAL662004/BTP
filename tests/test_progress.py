from src.utils.progress import Progress, text_bar


def test_text_bar():
    assert text_bar(0, 10, 10) == "░" * 10
    assert text_bar(5, 10, 10) == "█" * 5 + "░" * 5
    assert text_bar(20, 10, 10) == "█" * 10          # clamped
    assert text_bar(1, 0, 4) == "░" * 4              # total 0


def test_progress_iterates_everything_and_prints_redirected(capsys):
    out = list(Progress(range(10), desc="t", step_pct=50))
    assert out == list(range(10))
    err = capsys.readouterr().err                    # not a tty under pytest -> text lines
    assert "100%" in err and "█" in err and "t [" in err


def test_quiet_when_redirected(capsys):
    list(Progress(range(5), desc="q", quiet_when_redirected=True))
    assert capsys.readouterr().err == ""
