# Experiment 6 - Isaac

This is Isaac's part of experiment 6. The other work for experiment 6 can be found [here](https://docs.google.com/document/d/1iA6HhM_wVna6q9SUyQqsmNhnMgPEMotKMb8AKFq0Mbo/view?usp=sharing).

The experiment was about evaluating Mario's performance after training for the same amount of time at different resolutions. Lower resolutions train many more iterations with a compressed observation space. All resolutions were of the shape WIDTH x HEIGHT x 4, but the 4 (representing frame stacking) is ignored in this analysis. The findings are available [here](https://drive.google.com/drive/folders/1M4Q4egWkZoPO8647qhIPO8eMNXYkgatL).

The 240 x 256 resolution had the worst performance by a significant margin, and the 64 x 64 resolution and 128 x 128 resolution had similar performance. Overall, it seems that, at least with 1 hour of training on my device, there was not much difference between 64 x 64 and 128 x 128.
