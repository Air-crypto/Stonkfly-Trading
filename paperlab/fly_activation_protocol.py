"""Fixed future trading comparison motivated by the recipient stimulation assay."""
ARMS={
    'pristine_frozen':{'memory':'pristine','activity_reset':'carry','eta':.001,'recipient_current':0,'recipient_ids':['10704','11402']},
    'trained_frozen':{'memory':'paper_trained','activity_reset':'carry','eta':.001,'recipient_current':0,'recipient_ids':['10704','11402']},
    'pristine_stimulated':{'memory':'pristine','activity_reset':'carry','eta':.001,'recipient_current':10,'recipient_ids':['10704','11402']},
    'trained_stimulated':{'memory':'paper_trained','activity_reset':'carry','eta':.001,'recipient_current':10,'recipient_ids':['10704','11402']},
}
INFERENCE='Import only the pinned plastic weights/u/w for paper_trained arms; pristine arms use native initial memory. Start development and test separately with fresh dynamics and declared memory. Freeze all inference with no reinforcement. Carry activity between eligible observations. Apply the declared current to both identified MBON11 cells 10704 and 11402 throughout each 500 ms observation, recording exact native stimulation arguments; zero-current controls receive no extra current. Missing slots do not advance or reset the brain. Retain the native graph, kernel, all other parameters, and fixed decoder.'
SELECTION='Choose the higher development equity of trained_frozen and trained_stimulated; ties within 1e-9 prefer trained_frozen. Select it only if it strictly exceeds cash and both pristine controls by more than 1e-9. Persist selection before test simulation. Report every test arm without reselection or automatic policy deployment.'
RATIONALE='The prior controlled current assay reproduced all unstimulated counts. Current 5 left trained recipients silent; current 10 activated both memory conditions and exposed different decoded actions. Evaluate only the predeclared current 10 against zero on fresh market inputs; do not tune from these future outcomes.'
