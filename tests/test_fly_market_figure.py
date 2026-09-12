import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from paperlab.fly_market_figure import phase_series,render


@pytest.mark.parametrize("number", ["03", "04", "05"])
def test_market_figure_reconciles_each_panel_and_marks_gaps(number):
    report=json.loads(Path(f'reports/fly-market-study-{number}.json').read_text())
    for arm,totals in report['total_equity'].items():
        for phase,total in totals.items():
            times,equity,unavailable=phase_series(report,arm,phase)
            assert equity[-1]==total and len(times)==report['phase_steps']+1
    svg=render(report,'Replay <03>')
    ET.fromstring(svg)
    assert 'Replay &lt;03&gt;' in svg and '#ff8e8e' in svg
    assert 'No arm passed' in svg
    bad=copy.deepcopy(report)
    bad['total_equity']['pristine_frozen']['test']+=1
    with pytest.raises(ValueError,match='reconcile'):render(bad)
