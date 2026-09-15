from __future__ import annotations
from typing import Sequence
import numpy as np

class TemporalTrendAnalyzer:
    def analyze(self, years: Sequence[int], values: Sequence[float]) -> dict:
        if len(years) != len(values): raise ValueError("INVALID_TEMPORAL_PAIR")
        if len(values) < 3: return {"status":"insufficient_observations","observation_count":len(values),"slope":None}
        x=np.asarray(years,dtype=float); y=np.asarray(values,dtype=float)
        slope=float(np.polyfit(x,y,1)[0]); diffs=np.diff(y)
        direction="increasing" if slope>0 and np.all(diffs>=0) else "decreasing" if slope<0 and np.all(diffs<=0) else ("stable" if abs(slope)<1e-9 else "non_monotonic")
        return {"status":direction,"observation_count":len(values),"slope":slope,"absolute_change":float(y[-1]-y[0]),"relative_change":float((y[-1]-y[0])/abs(y[0])) if y[0] else None}
