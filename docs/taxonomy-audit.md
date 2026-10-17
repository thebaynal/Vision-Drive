# Why chair was excluded from primary evaluation

The initial experiment scored source Person and Chair annotations as COCO
person and chair. Figure QA exposed an important mismatch: **Open Images Stool
is separate from Chair**, while COCO chair predictions can legitimately cover
stools. Source Chair verification alone cannot guarantee completeness for the
broader COCO target concept.

For example, image **`3da6cd80dd411d74`** has verified **Chair=0, Stool=1** and
four source Stool boxes. The initial scorer counted chair predictions on the
stools as false positives. This is an annotation/taxonomy limitation, not
evidence that those model detections are necessarily wrong.

Only eight of the 249 cohort images have Stool verification: six positive
(two development, four test), two negative (one per partition). The remaining
241 are unknown. Removing only the known positives cannot establish complete
COCO-chair ground truth for the rest. No reliable ignored-region protocol can
be built from unknown regions, so **chair is excluded from primary scoring**.

The 249-image pilot and its intermediate person-only rescoring are not final
test evidence. A **fresh 400-image person-only cohort** excludes every pilot and
baseline image ID. Its protocol, source-only selection and image hashes were
frozen in commit `8a6f52a` before first inference. Confidence/NMS/matching
thresholds were retained; no fresh test score changed selection or configuration.

The earlier two-class measurements are preserved in
`docs/results/exploratory-person-chair/` as **exploratory, incomplete-taxonomy
results**. They must not be quoted as reliable chair performance or substituted
for the primary person result. Source annotations remain available for audit.

Future furniture evaluation requires either an exhaustively reannotated
COCO-compatible camera dataset or a source with verified coverage for the full
chair concept, including appropriate stool instances and reviewed ignored cases.
