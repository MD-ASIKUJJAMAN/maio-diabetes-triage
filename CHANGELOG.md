
# Changelog

## v0.1 - Baseline Release

### Model
- StandardScaler + LinearRegression
- Diabetes regression dataset
- 80% training and 20% testing
- Random seed: 42

### Performance
- Test RMSE: 53.85

### Features
- POST /predict endpoint
- GET /health endpoint
- HTTP 422 validation for missing features
- Docker container support
- Automated CI tests using GitHub Actions

### Limitations
- Uses an open demonstration dataset.
- Predictions are not clinically validated.
