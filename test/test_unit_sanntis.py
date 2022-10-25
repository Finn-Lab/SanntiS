from sanntis import _cli
import pytest

def testCliHelp():
    with pytest.raises(SystemExit) as e:
        _cli.main(['--help'])
    assert e.value.code == 0
