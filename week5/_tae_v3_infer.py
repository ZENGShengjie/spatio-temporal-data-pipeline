import sys
sys.path.insert(0, 'anomaly')
import numpy as np
import torch
from transformer_ae_v3 import TemporalAttentionAE, TAEV3Trainer, REC_CFG, cache_path

print('[TAE V3] Building model...')
trainer = TAEV3Trainer(target='taxi_flow_total')
trainer._preload()

# Build model and load weights
trainer.model = TemporalAttentionAE(
    seq_len=REC_CFG.seq_len,
    d_model=REC_CFG.hidden_dim,
    num_heads=REC_CFG.num_heads,
    dropout=REC_CFG.dropout,
).to(trainer.device)

ckpt = torch.load('cache/tae_weights_v3.pt.npy', map_location=trainer.device, weights_only=False)
trainer.model.load_state_dict(ckpt['state_dict'])
trainer.model.eval()
print('[TAE V3] Weights loaded.')

print('[TAE V3] Generating val scores...')
mask_val, scores_val = trainer.predict(split='val', use_topk=True, k=3, use_per_group_thresh=False)
print('[TAE V3] val scores: shape=' + str(scores_val.shape))

print('[TAE V3] Generating test scores...')
mask_test, scores_test = trainer.predict(split='test', use_topk=True, k=3, use_per_group_thresh=False)
print('[TAE V3] test scores: shape=' + str(scores_test.shape))

print('[TAE V3] Saving...')
np.save('cache/tae_scores_val_v3.npy', scores_val)
np.save('cache/tae_scores_test_v3.npy', scores_test)
np.save('cache/tae_mask_val_v3.npy', mask_val)
np.save('cache/tae_mask_test_v3.npy', mask_test)
print('[TAE V3] Done! val_detected=' + str(int(mask_val.sum())) + ' test_detected=' + str(int(mask_test.sum())))
