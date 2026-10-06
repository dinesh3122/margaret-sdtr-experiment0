# Experiment 0 flow

The notebook and scripted pipeline follow this order:

1. Load the configuration and verify CUDA, credentials, and upstream repositories.
2. Load the School of Reward Hacks data and construct disjoint training/calibration/final-evaluation splits.
3. Build matched SFT records for no-inoculation, inoculated, and honest-control arms.
4. Supply positive and negative reward-hacking calibration records.
5. Run the base model through Persona Vectors extraction and freeze prompt-last and response-average vectors.
6. Train the three LoRA treatment arms with matched optimization settings.
7. Save start, midpoint, and final checkpoints.
8. Collect prompt-last and response-average activations at the configured layers.
9. Supply the response-level reward-hacking evaluator.
10. Apply identical linear-steering sweeps to the base and treatment arms.
11. Save responses, evaluator scores, activations, projections, and run metadata.
12. Analyze whether behavior and internal projections support suppression, weaker formation, or generic SFT movement.

The final evaluation set must remain disjoint from training and all calibration sets. The package raises an error if it is omitted rather than silently reusing calibration examples.

