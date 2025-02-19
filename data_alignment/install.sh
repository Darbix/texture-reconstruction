#!/bin/bash

# # Step 0: Create and activate a virtual environment
# python3 -m venv venv
# source venv/bin/activate

# Step 1: Clone the repository
git clone https://github.com/princeton-vl/SEA-RAFT.git

# Step 2: Install dependencies from SEA-RAFT's requirements
python -m pip install -r SEA-RAFT/requirements.txt

# Step 3: Install dependencies from the current repository's requirements
python -m pip install -r requirements.txt

echo "Setup complete!"
