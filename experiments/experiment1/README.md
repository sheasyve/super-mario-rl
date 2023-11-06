# Experiment 1 (Isaac & Steven)

Our time was mostly spent setting things up, we didn't get to any proper reinforcement learning experiments. Here's what we did:

## Tasks

### Fix Shimmy Incompatibility

**Time:** ~3 hours.

This might be unnecessary since Shea supposedly had things working, but our setup wasn't functional due to an issue with specifying a seed for resetting the game environment not being compatible with JoypadSpace. We fixed it by adding a try/except block in `lib/Shimmy/shimmy/openai_gym_compatibility.py/GymV26CompatibilityV0/reset`. I think we spent roughly 3 hours on this.

### Docker Environment

**Time:** ~6 hours.

This is so we can train on anyone's PC. Unfortunately, graphics support isn't set up yet, so you'll have to change `render_mode` from `"human"` to `"rgb_array"` for now. I'm hoping to get this working at some point but no promises, this stuff is usually pretty finnicky.

### Preliminary Experiments

**Time:** ~3 hours.

We're running into some issues with performance, but we're trying to do experiments with normalising the input features and seeing how that affects training. Hopefully you'll make more ground than we did.

## Recommendations for Next Experiments

Getting things to run faster seems like a priority. Other than that, messing with the reward function, input features, action space and PPO hyperparameters are good bets for where to go next. I think at this stage we should probably focus more on reward function/input features/action space.
