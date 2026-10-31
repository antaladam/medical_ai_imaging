# Question and answer preparation — final presentation

100 questions with answers, based on `final_presentation.html` and `results.md` (final version,
2026-10-18). Difficulty: ★ basic term or fact · ★★ project understanding · ★★★ critical or
methodological.

---

## A. Terms and concepts (1–22)

**1. ★ What is a chest X-ray "finding"?**
A visible abnormality a radiologist describes in the report, such as an enlarged heart
(Cardiomegaly) or fluid around the lung (Pleural Effusion). The project scores six findings:
Lung Opacity, Cardiomegaly, Atelectasis, Lung Lesion, Pleural Effusion and Fracture.

**2. ★ What is the difference between a frontal and a lateral view?**
A frontal view is taken from the front or back (PA or AP) and shows both lungs side by side; a
lateral view is taken from the side. Every system in this project uses one frontal image per
study.

**3. ★ What is "ground truth"?**
The reference answer a system is scored against. Here it comes from the manual MeSH codes the
dataset's authors attached to each report, mapped to the six findings.

**4. ★ What are MeSH codes?**
Medical Subject Headings: a controlled vocabulary from the US National Library of Medicine. The
IU X-Ray authors coded each report with MeSH terms (e.g. "Pleural Effusion", "Opacity/lung"),
which makes them an independent, human-assigned label source.

**5. ★ What is sensitivity?**
Of the patients who really have the finding, the share the system catches. Sensitivity 0.62
means 62 of every 100 sick cases are found.

**6. ★ What is specificity?**
Of the patients who do not have the finding, the share the system correctly leaves alone.
Specificity 0.95 means 5 false alarms per 100 healthy cases.

**7. ★ What is precision?**
Of all the "yes" answers, the share that are correct. Unlike sensitivity and specificity, it
depends on how common the finding is.

**8. ★ What is F1?**
The harmonic mean of precision and sensitivity: 2 × P × S / (P + S). It is high only if both are
high, so neither "always yes" nor "almost never yes" scores well.

**9. ★ What does "macro" mean, as in macro AUROC or macro F1?**
The metric is computed for each of the six findings separately and then averaged, each finding
weighted equally. Rare findings such as Fracture count as much as common ones.

**10. ★ What is AUROC?**
Area under the ROC curve: the probability that a randomly chosen sick case gets a higher score
than a randomly chosen healthy case. 0.5 is chance, 1.0 is perfect ranking. It needs a
continuous score, not a yes/no answer, and does not depend on any cut-off.

**11. ★ What is a cut-off (threshold)?**
The value above which a score is turned into "yes". Each cut-off is a different trade-off: lower
catches more sick cases (sensitivity up) but raises more false alarms (specificity down).

**12. ★ What is Youden's J?**
J = sensitivity + specificity − 1. It is used to choose a cut-off: every candidate is tried and
the one with the highest J, the best balance of the two, is kept. It does not depend on how
common the finding is.

**13. ★ What is a confidence interval, and what is a bootstrap?**
A 95% confidence interval is the range in which the true value is likely to lie, given sampling
noise. The bootstrap estimates it by resampling the 400 test studies with replacement 2,000
times and recomputing the metric each time; the middle 95% of the results is the interval.

**14. ★★ What does "paired" bootstrap mean?**
For a difference such as G − S0, both systems are scored on the same resampled studies in each
round. That removes the variation that comes from which studies happen to be in the sample, so
the interval reflects the difference between the systems only.

**15. ★ What is an embedding?**
A list of numbers that represents an image or a text so that similar items get similar numbers.
BiomedCLIP turns each frontal X-ray into 512 numbers.

**16. ★ What is cosine similarity?**
A measure of how similar two embeddings are, from −1 to 1, based on the angle between them.
Retrieval returns the training images with the highest cosine similarity to the patient's image.

**17. ★ What is RAG (retrieval-augmented generation)?**
A pattern where relevant documents are retrieved first and given to a language model together
with the question, so its answer can draw on (and cite) that evidence. G is a RAG system: it
retrieves similar cases' reports before calling Gemini.

**18. ★ What is a zero-shot model?**
A model used on a dataset it was never trained on. S0 was trained on other hospitals' data
(MIMIC-CXR) and applied to IU X-Ray without any further training.

**19. ★ What is an "ablation"?**
A copy of a system with one part removed, to measure what that part contributes. G−ind removes
the clinical indication; G-img removes both the retrieved cases and the indication.

**20. ★ What is a "clinical indication"?**
The reason the X-ray was ordered, written by the referring doctor, e.g. "Dyspnea and edema". It
is the INDICATION section of each IU report.

**21. ★★ What does "calibrated" mean for probabilities?**
That they mean what they say: among cases given 0.8, about 80% really have the finding. G's
probabilities rank well but are overconfident on validation (stated ≈ 0.84 → present 35%).
Calibration was not part of this semester's evaluation.

**22. ★★ What is "label noise"?**
Errors or inconsistencies in the ground truth itself. Example: for CXR2130 the report mentions
two small nodules, but the MeSH codes do not include Lung Lesion, so "absent" counts as correct.

---

## B. The data (23–36)

**23. ★ What is IU X-Ray?**
The Indiana University chest X-ray collection, published by the US National Library of Medicine
through Open-i: 3,955 radiology reports with 7,470 images from two Indiana hospital systems,
de-identified, licensed CC BY-NC-ND 4.0 (Demner-Fushman et al., JAMIA 2016).

**24. ★ What does one study contain?**
An XML report with four sections (COMPARISON, INDICATION, FINDINGS, IMPRESSION), usually a frontal
and a lateral image, and the authors' manual MeSH codes.

**25. ★ What does "XXXX" in the reports mean?**
Text removed during de-identification (names, dates, sometimes ordinary words). It was stripped
before comparing texts in the memorisation check.

**26. ★★ How did 3,955 reports become 3,604 studies?**
104 reports have no image; 161 studies have only a lateral view; that leaves 3,690 with one
frontal image. 86 of those were never coded by the authors and have no ground truth, leaving
3,604. In addition, 4 unusable images (3 blank, 1 noise) were dropped without losing a study.

**27. ★★ How was the frontal image found, if the release has no view label?**
A zero-shot BiomedCLIP classifier compared each image with "frontal" and "lateral" text
prompts. Per study the most frontal image was kept if its frontal probability was ≥ 0.05. The
rule was checked by eye: 60/60 random images correct, all 11 low-score selections frontal, 40/40
excluded studies lateral.

**28. ★★ The release has no patient IDs. Why is a study-level split still safe?**
The dataset paper states that no more than one study per patient was taken. So each study is a
different patient, and a split by study is a split by patient: no patient can appear in both
the retrieval index and the test set.

**29. ★★ Why were the manual MeSH codes used instead of CheXbert, as planned?**
Checked against the MeSH codes, CheXbert labelled one templated normal report ("… Heart and
mediastinum normal.") as Cardiomegaly 55 times, 13% of all its Cardiomegaly positives, and no
preprocessing fixed it. The MeSH codes are an independent, human-assigned reference.

**30. ★ Why these six findings?**
They have enough positives in the collection, and the specialist model S0 has an output for each
of them. Support Devices had enough positives but no S0 output, so it was dropped.

**31. ★ How common are the findings?**
Lung Opacity 519 (14.4%), Atelectasis 296, Cardiomegaly 295, Pleural Effusion 138, Lung Lesion
104, Fracture 83 (2.3%). 2,623 of 3,604 studies (72.8%) have none of the six.

**32. ★ How was the data split?**
Train 2,804 studies (the only retrieval index; S0 and R0 cut-offs), validation 400 (natural
prevalence; G's cut-off; includes the 50-study pilot), test 400 (used once).

**33. ★★ Why is the test set "enriched", and what does it change?**
Positives were sampled until every finding had at least 35 (filled rarest first), so rare
findings like Fracture (38) can be measured. Sensitivity, specificity and AUROC are unaffected
by this; F1 and precision are higher than at natural prevalence and should only be compared
within the test set.

**34. ★★ Why were G's cut-offs not tuned per finding on validation?**
Validation has natural prevalence and only 2 Fracture, 11 Lung Lesion and 12 Pleural Effusion
positives, too few for a reliable per-finding cut-off. So one cut-off was pooled over all six
findings (2,400 answers, 113 positives).

**35. ★★ What is the pilot set used for?**
50 validation studies for prompt development and behaviour checks (e.g. P1's failed calls, P3's
truncated answers, G's prompt versions). It is never used to report accuracy.

**36. ★★★ Could near-duplicate images leak between splits?**
Checked twice: BiomedCLIP similarity (no pair reaches 0.99; the 30 most similar pairs were
inspected and are different patients) and pixel level (no identical files or thumbnails). No
duplicates were found.

---

## C. The reference systems S0 and R0 (37–46)

**37. ★ What is S0?**
The specialist image model: a DenseNet-121 from the TorchXRayVision library, trained on MIMIC-CXR
(Boston). It reads the frontal image alone and outputs a score per finding. No language model,
no text, no retrieval.

**38. ★★ Walk through S0's pipeline.**
Frontal image → pad to a square with black borders → resize to 224 px → DenseNet-121 → a score
per finding → per-finding cut-off chosen with Youden's J on the train split → yes/no.

**39. ★★ Why padding instead of the library's centre crop?**
Many IU films are landscape; the centre crop cut off the chest edges. Padding keeps the whole
image and improved Pleural Effusion and Fracture, which sit near the edges.

**40. ★★ Why were S0's cut-offs tuned on train and not on validation?**
S0 was never trained on IU data, so the train split is unseen for it, and train has enough
positives per finding. Validation has too few rare positives.

**41. ★★ Why are S0's cut-offs so low (e.g. 0.007 for Lung Lesion)?**
Youden's J favours sensitivity when a score separates cases poorly; S0 is near chance on Lung
Lesion, so the best balance sits at a cut-off that calls most studies positive. On the example
study CXR2130, S0 says "yes" to all six findings.

**42. ★★★ Why was the weaker DenseNet kept when a ResNet-50 scored 0.808 vs. 0.728?**
The frozen plan names a DenseNet, and the ResNet's Lung Lesion output is untrained (it would need
a proxy). The decision was to stay on the plan and report the ResNet as a reference only; it was
never run on test. The consequence is stated as a limit: S0 is the planned specialist, not the
best available.

**43. ★ What is R0?**
Retrieval alone, no language model: BiomedCLIP finds the k most similar training images, and the
score for each finding is the share of those neighbours that have it.

**44. ★★ How was k chosen for R0, and why 150?**
By leave-one-out on train: each train study was scored from the other 2,803, for k from 1 to
1,000. Macro AUROC rises to about k = 100 and then plateaus; k = 150 was highest (0.741).

**45. ★★ What does R0's k-curve say about retrieval?**
Single neighbours carry little signal (k = 1: 0.572); the useful signal is the prevalence of a
finding in a broad neighbourhood. This is why P2's top-1 retrieval is weak, and why G uses 10
cases rather than 1.

**46. ★★★ Why does R0 matter for interpreting G?**
G uses the same index, so R0's first 10 neighbours are exactly G's 10. G − R0 asks whether the
language model adds anything beyond retrieval: +0.028 macro AUROC, interval [−0.004, +0.060],
which includes zero, so it is not shown.

---

## D. The starter apps (47–62)

**47. ★ What are P1, P2 and P3?**
The three assigned starter apps: P1 = `ai_medical_imaging_agent`, P2 = `vision_rag`, P3 =
`multimodal_agentic_rag`, all tutorial apps from the awesome-llm-apps collection.

**48. ★ What does P1 do as shipped?**
A Streamlit app: the user uploads a medical image and an agno agent with Gemini 2.5 Pro writes a
structured report following a fixed five-section prompt, with a DuckDuckGo web-search tool.

**49. ★ What are P1's five prompt sections?**
1 Image Type & Region, 2 Key Findings, 3 Diagnostic Assessment, 4 Patient-Friendly Explanation,
5 Research Context (use DuckDuckGo to find literature, treatment protocols, links and 2–3
references).

**50. ★★ Why was section 5 removed (P1-adapted-ns)?**
Web search was turned off because live search is not reproducible. The prompt still ordered the
model to use the search tool, so it tried to call a tool that did not exist and 62% of pilot
calls failed (MALFORMED_FUNCTION_CALL). Removing the section fixed this: 50/50 answered.

**51. ★ What does P2 do as shipped?**
A Streamlit app: the user's images are embedded with Cohere embed-v4; a text question is
embedded, the single most similar image is retrieved (top-1), and that image plus the question
go to Gemini 2.5 Flash.

**52. ★★ Why was the patient's own image added to P2?**
As shipped, only the retrieved image reaches the model. Run on a patient, it would describe a
different patient's X-ray. So the search uses the patient's image and both images are sent.

**53. ★★ What else changed in P2?**
Gemini Embedding 2 instead of Cohere (no Cohere account), gemini-3-flash-preview instead of the
retired Gemini 2.5 Flash, and a batch runner instead of the UI.

**54. ★ What does P3 do as shipped?**
A React + FastAPI app over a multimodal knowledge base: documents are embedded with Gemini
Embedding 2 (image blended 0.68 / 0.32 with its annotation), and a Google ADK agent answers from
the top 6 matches with citations.

**55. ★★ What is P3's key limitation for this task?**
Its generator only ever sees the retrieved documents' text, never an image. So it answers about
its neighbours, not about the patient: on CXR2130 it reported the neighbours' atelectasis and
granuloma as the patient's findings.

**56. ★★ Why was P3's output cap removed (P3-adapted)?**
The shipped 900-token cap includes the model's thinking tokens; with the checklist question it
cut 45 of 50 pilot checklists. The cap was removed, as for P1 and P2; nothing else changed.

**57. ★ Why do all three starters run on gemini-3-flash-preview?**
The original models (Gemini 2.5 Pro, 2.5 Flash) are retired for new users. Using one model also
removes model differences between the starters.

**58. ★★ What is the "checklist question", and why was it needed?**
The fixed question asks the system to begin with six lines "<finding>: present / absent /
uncertain". It was added because the planned labeller, CheXbert, misread the starters' free text,
e.g. "Cardiomegaly: Not present" → positive. The six lines are parsed directly instead.

**59. ★★ How are the starters' answers scored?**
"present" counts as yes; "absent", "uncertain" and missing count as no. On test, all 1,200
checklists were parsable. A hand check of 45 answers found the checklist agreed with the rest of
the answer in 268 of 270 findings.

**60. ★★ Why do the starters have no AUROC?**
They give only verdicts (present / absent / uncertain), no score per finding. AUROC needs a
continuous score, so they are compared on sensitivity, specificity and F1.

**61. ★ How did the starters do on test?**
Macro sensitivity / specificity / F1: P1-adapted-ns 0.52 / 0.86 / 0.42, P2-adapted 0.42 / 0.90 /
0.39, P3-adapted 0.43 / 0.75 / 0.30.

**62. ★★★ The starters were changed. Can they still be called "the starters"?**
Each change was the minimum needed to run them on chest X-rays and is recorded with its reason;
arms with a core change are renamed (P1-adapted-ns, P2-adapted, P3-adapted). They are not
exactly as shipped, which is stated as a limitation; the Q1 answer is precisely that none ran on
chest X-rays out of the box.

---

## E. The built system G (63–77)

**63. ★ What is G?**
The built system: the patient's frontal image + the clinical indication + the reports of the 10
most similar training cases → one Gemini call → for each finding a status, a probability, cited
case IDs and a rationale, plus a short impression.

**64. ★★ Walk through G step by step.**
1 Embed the patient's image with BiomedCLIP; 2 retrieve the 10 most similar train studies (R0's
index); 3 assemble the prompt (finding definitions, how to work, the indication, the 10 reports,
the image); 4 one gemini-3-flash-preview call with an enforced JSON schema, no tools; 5 validate:
exactly six findings, probabilities in 0–1, every cited ID among the 10 given.

**65. ★★ What did G take from each starter?**
From P1: a structured, systematic reading of the patient's image, made precise with finding
definitions. From P2: embed → retrieve → generate with the patient's image, upgraded to
BiomedCLIP and 10 cases. From P3: retrieved reports as evidence and citations, made per finding
and checkable, with the patient's image added.

**66. ★★ What does G add that no starter has?**
The clinical indication as input, a probability per finding (needed for AUROC), the patient's
image and textual evidence together, definitions aligned with the ground truth, and citations
that code can verify.

**67. ★★ Why BiomedCLIP rather than a general-purpose embedder?**
In the project's own comparison (Q2), BiomedCLIP retrieved cases with matching findings better
than Gemini Embedding at small k. It is trained on biomedical images and captions.

**68. ★★ Why k = 10 for G?**
Fixed in advance from R0's curve: k = 5 gives 0.651, k = 10 0.671, k = 20 0.704 (more neighbours
carry more signal), while 10 reports still fit easily in one prompt. Tuning k on validation was
not done.

**69. ★★ Why one direct call instead of an agent?**
The agent-based starters showed hidden retries, failed tool calls and runaway answers. A single
call with a fixed prompt and schema is simpler, reproducible and cheaper to check.

**70. ★★ Where in the prompt is the probability requested?**
Step 4: "give its status (present, absent or uncertain), the probability (0 to 1) that it is
present in this patient, …" and "Give the probability with two decimals as your honest estimate;
for subtle, borderline or partly obscured appearances, use intermediate values …". The JSON schema
also requires a `probability` field for each finding.

**71. ★★ Why were there two prompt versions?**
On the 50 pilot studies, prompt 1 led to citations by list position (3/50) and to coarse
probabilities (198 of 300 exactly 0.05). Prompt 2 asks for citation by ID and two-decimal
probabilities; it was then frozen and used for validation and test.

**72. ★★ What are G−ind and G-img?**
G−ind is G without the indication line; G-img keeps G's model, prompt and output format but gets
only the image (no cases, no indication). G vs. G−ind measures the text; G vs. G-img measures
cases + text together.

**73. ★★ How was G's cut-off 0.08 obtained?**
On G's answers for the 400 validation studies, every probability value was tried as a cut-off,
pooled over the six findings (2,400 answers). Each was scored with Youden's J; the highest was at
0.08 (pooled sensitivity 0.64, specificity 0.82). It was fixed before the test run; G−ind's also
came out at 0.08.

**74. ★★★ G gives a status and a probability. Which is "its answer"?**
Both are used, for different comparisons. The status (only "present" = yes) is compared with the
starters, which also give only verdicts. The probability is used for AUROC against S0 and R0, and
for yes/no at the 0.08 cut-off. On G's typical "absent, 0.10", the two readings disagree.

**75. ★★★ Why does G's stated "present" have sensitivity 0.32 but the cut-off 0.62?**
G's own "present" behaves like a cut-off near 0.5: very cautious (validation 0.30 / 0.97). At
0.08 the probabilities catch about twice as many sick cases while keeping specificity at 0.78 on
test.

**76. ★★ What did G-img's "invented citations" show?**
G-img was given no cases but kept G's instruction to cite cases and the citation field in the
schema. In 2 of 450 answers it filled the field with case IDs it was never given. The validator
flagged them. An output slot for evidence can elicit fabricated evidence when none exists.

**77. ★★ How is a citation checked?**
Automatically: every cited ID must be one of the 10 cases given to G for that study. Then, for
evaluation, the cited case's own labels are compared with the finding G claimed.

---

## F. Evaluation and results (78–92)

**78. ★ What does "run once on the test set" mean, and why does it matter?**
Every system was run on the 400 test studies exactly once, after all prompts, cut-offs and
methods were fixed. Looking at test results and then adjusting would fit the systems to this one
sample and overstate how well they work.

**79. ★★ How was accidental use of the test set prevented?**
Every script that can touch the test split refuses unless it is called with an explicit `--test`
flag, and all choices were fixed on train and validation beforehand.

**80. ★★ What were the pre-declared comparisons?**
#1 (primary) G − S0, macro AUROC; #2 G − R0; #3 G − G−ind; #4 G − G-img; #5 G-img − S0; #6–8 G
vs. each starter on sensitivity, specificity and F1. They were fixed before the test run so the
answer could not be picked afterwards.

**81. ★★ Why is G − S0 on macro AUROC the primary comparison?**
It is the thesis question (does image + retrieval + text beat a single image model?), and AUROC
compares the two systems' ranking without depending on either system's cut-off.

**82. ★ What is the main result?**
G's macro AUROC is 0.754, S0's 0.705: a difference of +0.049, 95% CI [+0.013, +0.087]. The
interval excludes zero, so on the pre-declared criterion G ranks findings better.

**83. ★★ On which findings does G gain over S0?**
Mainly Lung Lesion (+0.19) and Cardiomegaly (+0.07); on Lung Opacity, Atelectasis, Pleural
Effusion and Fracture the two are level.

**84. ★★★ What happens to the main result without Lung Lesion?**
G − S0 drops to +0.020 [−0.020, +0.063], and to +0.008 without Lung Lesion and Fracture:
not distinguishable from zero. The gain rests largely on one finding, on which S0 is near chance
in every split. Why S0 is so weak there is not established. It is not a different definition: S0's
training labels (MIMIC-CXR, extracted by NegBio with the CheXpert labeller's phrases) count a
nodule, mass or cavity and exclude calcified nodules, essentially as the ground truth does. Likely
candidates are noisy text-mined training labels, small lesions at 224 px and poor transfer between
hospitals. This check was promised before the test run.

**85. ★★★ Why is this not "moving the goalposts" after seeing the results?**
The opposite: the check was written down on 2026-10-17, before any test run, as "macro averages
are also reported without them if it changes a conclusion". The pre-declared result stands as
declared; the sensitivity check is reported next to it.

**86. ★★ Where does G's advantage come from?**
Gemini reading the image alone (G-img) is level with S0: +0.010 [−0.029, +0.050]. Adding the
cases and the indication gives +0.039 [+0.008, +0.071]. The two add up to the primary difference
(0.010 + 0.039 = 0.049).

**87. ★★★ Which result is the most robust?**
G − G-img: what the cases and indication add within one design. It holds without Lung Lesion
(+0.040 [+0.005, +0.077]), whereas G − S0 does not.

**88. ★★ Did the clinical indication help on its own?**
Not shown: G − G−ind is +0.020 [−0.008, +0.046]. The interval includes zero. Together with the
cases (G − G-img) the gain is clear.

**89. ★★ How does G compare with the starters?**
G's stated verdicts are more cautious, not more accurate: lower sensitivity than every starter
(−0.09 to −0.20), higher specificity (+0.05 to +0.20). On F1 it ties P1 and P2 and beats only P3
(+0.097 [+0.040, +0.149]).

**90. ★★ What are G's advantages, then?**
A probability per finding that ranks well (AUROC), and citations that can be checked: 81–96% of
the cases it cites for a "present" call have that finding, against 24–43% of all the cases it
was shown (Lung Lesion is weaker: 43% vs. 7%).

**91. ★★★ Does G simply copy its neighbours?**
Mostly not. Where the neighbours' vote is wrong (524 of 2,400 pairs), G agrees with it 43% of the
time and G-img, which never saw them, 40%: a pull of about 3 points. When G does copy, its
rationale says so ("reinforced by its presence in the majority of matched similar cases").

**92. ★★ How good is the retrieval?**
G's 10 neighbours contain the patient's finding 1.2 to 7.2 times as often as random cases:
strongest for Pleural Effusion (×7.2), weakest for Fracture (×1.2).

---

## G. Analysis, limitations and critical questions (93–100)

**93. ★★ What kinds of errors did the failure review find?**
Of 23 sampled errors: missed findings 6 (mainly old rib fractures), copied neighbours 6, invented
findings 4 (e.g. an "enlarged heart" on low-volume films), definition mismatches 3 (nodules or
scarring vs. Lung Opacity), other 4 (cut-off effects, a misclassified abnormality, an invented
citation).

**94. ★★★ Could Gemini have memorised IU X-Ray?**
Probably it has seen the dataset. The check compared each system's text with the study's own
report and with 20 reports carrying the same labels: no system matched its own report beyond
the baseline in word 3-grams (all intervals include zero; own report beats all 20 in 4.9–6.8% of
studies vs. 5% by chance), including the image-only G-img. Recalled labels without recalled
wording cannot be excluded.

**95. ★★ Do G's answers match its own reasoning?**
In a hand check of 15 answers, the status agreed with the rationale in 88 of 90 findings (2
unclear, 0 contradictions). All 17 "present" calls were stated in terms of the image; cases or
the indication appeared only as support. Rationales are self-reports, not proof of how the model
decided.

**96. ★★★ What are the main limits of the main claim?**
S0 is the planned DenseNet, not the strongest specialist; the gain rests on Lung Lesion, where
S0 is near chance; Gemini has probably seen IU X-Ray; G gets finding definitions the other
systems do not; S0 is zero-shot. The claim is "G beats the planned zero-shot specialist on this
collection, mainly on one finding", not "multimodal retrieval beats image models".

**97. ★★★ What are the limits of the study design?**
One hospital collection and one 400-study test sample (wide intervals for rare findings); no
external validation; an enriched test set (F1 not comparable with natural prevalence); hand
judgements by an AI assistant from text only, with no radiologist; starters adapted, not exactly
as shipped.

**98. ★★★ Would a fine-tuned or stronger image model change the conclusion?**
Possibly. On train, the ResNet-50 was 0.080 higher than the DenseNet; that is a different split,
but a larger margin than G's 0.049 lead on test. S0 was also never trained on IU data. The project does not claim G beats the best image
model; a fine-tuned specialist is part of the thesis plan.

**99. ★★★ If you could choose G's cut-off freely, would you pick 0.08?**
Any point is a trade-off: 0.15 gives about 0.43 sensitivity / 0.92 specificity on validation,
0.05 collapses specificity to 0.22 because 1,399 of 2,400 answers are exactly 0.05. 0.08 is the
best balance by Youden's J, and it had to be fixed on validation before the test run, since
choosing it after seeing test results would be peeking. AUROC, the primary measure, does not
depend on the cut-off at all.

**100. ★★ What are the answers to Q1, Q2 and Q3 in one sentence each?**
Q1: none of the starters ran on chest X-rays as shipped; adapted, they answer every study, but
their verdicts are coarse and P3 never reads the patient's image. Q2: G combines P1's systematic
reading, P2's retrieval and P3's citations, and adds the indication, a probability per finding
and machine-checkable citations. Q3: G beats the planned specialist (+0.049 macro AUROC), but
mostly on Lung Lesion; the robust result is that cases + context make Gemini better than Gemini
on the image alone.
