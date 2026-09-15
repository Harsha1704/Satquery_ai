"""Deterministic temporal QA over computed facts only."""
from __future__ import annotations
class TemporalChangeVQA:
 def answer(self, query:str, result:dict)->dict:
  q=query.lower(); stats=result["statistics"]; dominant=result.get("dominant_transition") or "No semantic transition was available"
  if "how much" in q or "area changed" in q: answer=f"{stats['changed_area_ha']:.4f} hectares ({stats['changed_percentage']:.2f}% of the effective analysis area) changed."
  elif "where" in q: answer=f"Most mapped change is in the {result.get('largest_change_direction','unknown')} portion of the effective ROI."
  elif "dominant" in q or "transition" in q: answer=f"The dominant semantic transition was {dominant}."
  elif "unchanged" in q: answer=f"{stats['unchanged_percentage']:.2f}% of valid effective-ROI pixels remained unchanged."
  else: answer=result["description"]
  return {"answer":answer,"evidence_result_id":result["result_id"],"grounded":True}
