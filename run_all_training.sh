#!/bin/bash

uv run greedy.py
uv run epsilon_greedy.py
uv run decaying_epsilon_greedy.py
uv run ucb1.py
uv run thompson_sampling.py
uv run gittins_index.py

