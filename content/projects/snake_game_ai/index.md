+++
date = '2025-02-10'
draft = false
title = 'AI Snake Game Bot'
+++

Repo: [Github – SnakeGameAI](https://github.com/sihanyu03/SnakeGameAI)

I built a simple snake game and trained an AI to play it using NEAT (NeuroEvolution of
Augmenting Topologies) machine learning. Demo below:

{{< video src="snake_game_demo.mp4" >}}

---

## Overview

The agent operates on a deliberately minimal input configuration: binary indicators for
obstacles directly ahead and to either side, and the angle to the food. More complex state
representations were also explored, but they tend to lead to more ineffective training.

Despite the limited information, the agent learns competent behaviour. It can navigate
around the map well, avoiding additional obstacles I added, as well as its own
body, especially during early and mid-game. However, as the snake has no global knowledge,
it may trap itself in dead ends, which is what happened in the demo video as well. This
becomes increasingly likely as the snake grows longer.

This project primarily served as an exploration of the strengths and limitations of NEAT.
In hindsight, a classical path-finding algorithm like depth-first search, extended to
consider its own body and its future position, would most likely outperform this and
be simpler to implement. Nevertheless, this project was a fun investigation of
NEAT, and highlights the importance of finding an appropriate approach for a given problem.