# Mario Reinforcement Learning

This is our group semester project for SENG 474: Data Mining. Our goal is to explore reinforcement learning, and we're doing it with Super Mario Bros in Python.

## Repository Navigation

- `docker`: This is for our docker setup so we don't all have to configure everything ourselves. Right now it's working with `render_mode="rgb_array"`, but graphics aren't supported yet.
- `lib`: External software that we've modified for our use. Right now it's just some minor modifications we've made to shimmy.
- `Makefile`: Convenience commands, primarily for docker.
- `requirements.txt`: The Python libraries we're using. You can install them with `pip -r requirements.txt`.
- `src`: This is where our code lives.

## Docker Environment Instructions

1. [Install Docker](https://docs.docker.com/get-docker/).
2. Install Make:
   - **Windows:** You have (at least) 2 options.
     - [Download from GNUWin32](https://gnuwin32.sourceforge.net/packages/make.htm), or
     - if you have chocolatey, you can run `choco install make`.
   - **MacOS:** Apparently `xcode-select --install` works. I wouldn't know, I'm not a mac user.
3. Launch the environment by navigating to the repository root and running `make buc`.
4. When you're done working in the environment, run `exit` to leave it. Then, run `make down` from the repository root to shut down the environment. _If you don't shut the environment down, it will run in the background and consume resources. The only reason you want this is if you're currently running experiments in the background._
