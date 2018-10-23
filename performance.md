# Performance record

## Undirected Graph

- 3-layer GCN: Train Loss: 0.751895, Validation Loss: 0.612020
- 4-layer GCN: Train Loss: 0.744656, Validation Loss: 0.611218
- 4-layer, circle: Train Loss: 0.757772, Validation Loss: 0.616607
- 4-layer, linear time decay, circle: Train Loss: 0.725167, Validation Loss: 0.594566

## Directed Graph

- 4-layer, linear time decay:                           Train Loss: 1.446856, Validation Loss: 1.161222
- 4-layer, linear time decay, circle:                   Train Loss: 0.722132, Validation Loss: 0.618965
- 4-layer, linear time decay, circle, weak reverse=0.01: Train Loss: 0.757974, Validation Loss: 0.538613
- 4-layer, linear time decay, circle, weak reverse=0.1: Train Loss: 0.692573, Validation Loss: 0.525955
- 4-layer, linear time decay, circle, weak reverse=0.5: Train Loss: 0.715632, Validation Loss: 0.528355
