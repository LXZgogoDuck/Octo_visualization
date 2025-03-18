#!/bin/bash

# List of remote servers
servers=(
  linux hercules dgx1 savitar legolas bilbo gandalf sauron gollum gimli saruman
  orion hydra phoenix tucana lynx libra antlia phoenix leo
)

# Iterate through each server and copy the SSH key
for host in "${servers[@]}"; do
  echo "Copying SSH key to $host.cs.ox.ac.uk..."
  ssh "vis24xl@${host}.cs.ox.ac.uk" "mkdir -p ~/.ssh && echo '$(cat ~/.ssh/id_ed25519.pub)' >> ~/.ssh/authorized_keys"
done

echo "SSH key copy process completed!"

