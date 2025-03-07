from tools import printProgressBar


def test_printProgressBar():
    result = printProgressBar(3, 5, '', '', 1, 10)
    assert result is None
