# Understand the project before presenting it

## Explain the experiment in one minute

“I compared a CNN trained from scratch with frozen and fine-tuned MobileNetV2 models on six waste-image classes. I removed conflicting duplicates, grouped similar images before splitting, and selected the final model using validation macro-F1. I reported held-out results, inspected mistakes and built a local image-upload demo. Independent phone-photo evaluation is a separate workflow and remains pending until I collect the images.”

Replace the final sentence only when that evaluation has actually been performed.

## Questions to practice

- **Why transfer learning?** ImageNet weights provide reusable image features. The new head learns class boundaries; fine-tuning adapts the final feature blocks with a smaller learning rate.
- **Why macro-F1?** It weights each class equally, making failure on the small trash class more visible than overall accuracy alone. Always inspect per-class support and recall too.
- **Why three partitions?** Training updates weights, validation selects checkpoints/configurations, and test estimates performance after those choices are frozen.
- **Why group duplicates?** Seeing nearly identical examples in training and test can exaggerate generalization. Perceptual hashes help, but are imperfect physical-object identifiers.
- **Why keep frozen BatchNorm in evaluation mode?** Disabling gradients alone does not stop its running statistics from changing.
- **How is fine-tuning different from training the head?** Feature blocks 14 onward receive gradients; earlier blocks remain frozen. The backbone learning rate is lower than the head learning rate.
- **Why can a confident prediction be wrong?** Softmax scores compare known classes and are not calibrated correctness estimates. Unfamiliar objects may still receive high scores.
- **What does a false prediction teach you?** Inspect lighting, crop, background, mixed materials and label ambiguity. Keep the test set out of further tuning.
- **What is still missing?** Independent phone-photo collection, multiple-seed robustness estimates and broader deployment validation.

## Practical exercises

1. Read the split audit and explain the six exclusions and candidate near-duplicate groups.
2. Compare per-class scores with the confusion matrix; describe at least five mistakes.
3. Explain why the validation winner is the deployed model even if another model has a better test score.
4. Try your own image and record its prediction, then explain why a single example proves little.
5. Reproduce training in a fresh run with a predeclared seed. Never claim improvements until measured under a valid evaluation protocol.
