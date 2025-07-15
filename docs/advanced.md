# Advanced Usage

## Working with Different Input Types

### Multi-Channel Inputs

```python
# Works with multi-channel inputs
batch_size = 2
channels = 3
height = 64
width = 64

pred = torch.randn(batch_size, channels, height, width)
target = torch.randn(batch_size, channels, height, width)

loss_fn = WaveletLoss(wavelet="db4", level=2)
losses, details = loss_fn(pred, target)
```

## Handling Return Values

The loss function returns two values:
1. A list of losses for each level
2. A dictionary of detailed transform information

```python
losses, details = loss_fn(pred, target)

# Accessing individual level losses
total_loss = sum(losses)

# Examining transform details
print(details.keys())
# Typically includes: 'combined_hf_pred', 'combined_hf_target'
```

## Performance Considerations

- Choose appropriate wavelet and level based on your data
- Consider computational overhead of different transform types
- QWT is most computationally intensive
- DWT is typically the most efficient