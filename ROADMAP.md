# SRODecoderRing - Project Roadmap

This document outlines planned improvements and development priorities for the SRODecoderRing project.

## Current Status

The project currently has:
- ✅ Working Energy-Based Model (EBM) for rotation order inference
- ✅ PyTorch-based training pipeline
- ✅ F# production inference engine
- ✅ Flask REST API deployed on Azure
- ✅ Elmish web frontend on IPFS
- ✅ WPF desktop application

## Known Issues (Immediate Attention)

### High Priority
1. **Site Association** (`README.md:182`)
   - Multi-site treatment planning needs refinement
   - Current implementation may not handle complex multi-site cases correctly
   - Impact: Patient treatment accuracy

2. **iView Field Association** (`README.md:183`)
   - Issue: Applying shifts after first field with TPO that won't associate with second field
   - Impact: Multi-field treatment workflows
   - Needs investigation and resolution

## Short-Term Improvements (Q1-Q2 2025)

### Documentation & Developer Experience
- [ ] **API Documentation**
  - Add OpenAPI/Swagger spec for Flask REST API
  - Document request/response formats
  - Provide example API calls with curl/Python/JavaScript

- [ ] **Training Guide**
  - Step-by-step tutorial for training new models
  - Hyperparameter tuning guide
  - Dataset preparation best practices

- [ ] **Deployment Guide**
  - Document IPFS deployment process
  - Azure deployment automation
  - Docker containerization for Flask API

- [ ] **Code Documentation**
  - Add docstrings to Python modules (numpy/Google style)
  - XML documentation comments for C#/F# code
  - Inline comments for complex algorithms

### Testing & Quality Assurance
- [ ] **Expand Test Coverage**
  - Python unit tests for EBM components
  - Integration tests for API endpoints
  - UI end-to-end tests with Selenium/Playwright

- [ ] **Continuous Integration**
  - GitHub Actions workflow for automated testing
  - Automated build verification for .NET components
  - Python linting (ruff, mypy, black)

- [ ] **Clinical Validation Suite**
  - Known test cases from clinical data
  - Regression test suite
  - Performance benchmarks

### Model Improvements
- [ ] **Model Performance**
  - Benchmark inference time (target: <100ms)
  - Optimize MCMC sampling efficiency
  - Investigate model quantization for faster inference

- [ ] **Training Infrastructure**
  - Add training metrics dashboard (TensorBoard/Weights & Biases)
  - Automated hyperparameter search
  - Model versioning and experiment tracking

## Medium-Term Enhancements (Q3-Q4 2025)

### Feature Development
- [ ] **Enhanced Multi-Field Support**
  - Resolve iView field association issue
  - Support complex multi-field treatment plans
  - Handle field-specific rotation orders

- [ ] **Site Association Improvements**
  - Implement robust multi-site treatment planning
  - Add site-specific coordinate system handling
  - Validate against clinical scenarios

- [ ] **Uncertainty Quantification**
  - Provide confidence scores for predictions
  - Alert when model is uncertain
  - Human-in-the-loop review for low-confidence cases

- [ ] **Real DICOM Integration**
  - Direct DICOM SRO parsing
  - Integration with DICOM network (DIMSE)
  - Support for DICOM worklist queries

### Architecture Enhancements
- [ ] **Microservices Architecture**
  - Separate training service
  - Dedicated inference service
  - Model registry service

- [ ] **Caching Layer**
  - Redis cache for frequent queries
  - Model weight caching
  - Response caching with invalidation

- [ ] **Observability**
  - Structured logging (JSON)
  - Distributed tracing (OpenTelemetry)
  - Metrics and alerting (Prometheus/Grafana)

### User Experience
- [ ] **Web UI Improvements**
  - Batch processing interface
  - Historical query visualization
  - Export results to CSV/Excel

- [ ] **Desktop UI Enhancements**
  - Real-time visualization of transformations
  - Interactive 3D coordinate system viewer
  - Drag-and-drop DICOM file support

- [ ] **Mobile Support**
  - Responsive web design
  - Progressive Web App (PWA)
  - Offline capability

## Long-Term Vision (2026+)

### Advanced ML Capabilities
- [ ] **Active Learning**
  - Collect user feedback on predictions
  - Retrain model with corrected examples
  - Continuous model improvement

- [ ] **Multi-Task Learning**
  - Predict rotation order + patient position
  - Joint learning of coordinate systems
  - Transfer learning from related tasks

- [ ] **Explainable AI**
  - Attention visualization
  - SHAP values for interpretability
  - Human-readable explanations

### Clinical Integration
- [ ] **Treatment Planning System Integration**
  - Plugin for Eclipse/RayStation/Pinnacle
  - Native integration APIs
  - Real-time validation during planning

- [ ] **Clinical Decision Support**
  - Alert system for unusual transformations
  - Automated quality assurance checks
  - Integration with clinical workflows

- [ ] **Regulatory Compliance**
  - FDA 510(k) preparation (if pursuing medical device status)
  - IEC 62304 software lifecycle compliance
  - Clinical validation studies

### Research & Innovation
- [ ] **Lie Group Mathematics**
  - Leverage SO(3) structure explicitly
  - Geometric deep learning approaches
  - Equivariant neural networks

- [ ] **Federated Learning**
  - Train on distributed clinical datasets
  - Privacy-preserving learning
  - Multi-institutional collaboration

- [ ] **Benchmarking**
  - Public dataset for rotation order prediction
  - Comparison with analytical methods
  - Published research paper

### Infrastructure
- [ ] **Multi-Cloud Support**
  - Support AWS, GCP in addition to Azure
  - Kubernetes deployment
  - Auto-scaling inference

- [ ] **Edge Deployment**
  - ONNX export for edge devices
  - TensorRT optimization
  - Embedded systems support

## Migration from Keras to PyTorch

**Status**: ✅ Complete

The project has migrated from Keras to PyTorch 2.0.1+. Legacy references in code/docs should be updated:
- [x] Core training pipeline (completed)
- [ ] Remove Keras references from comments
- [ ] Update any remaining Keras-based notebooks

## Performance Targets

| Metric | Current | Target (2025) | Target (2026) |
|--------|---------|---------------|---------------|
| Inference Time | ~200ms | <100ms | <50ms |
| Model Accuracy | ~95% | >98% | >99.5% |
| API Availability | 99% | 99.9% | 99.95% |
| Test Coverage | ~60% | >80% | >90% |

## Contributing to the Roadmap

Have ideas for improvements? Here's how to contribute:

1. **Open an Issue**: Describe the feature or improvement
2. **Discuss**: Engage with maintainers and community
3. **Prioritize**: Help us understand impact and urgency
4. **Implement**: Submit a PR once approved

## Timeline Summary

```
2025 Q1-Q2: Documentation, Testing, CI/CD
    │
    ├─ API docs & deployment guides
    ├─ Expanded test coverage
    └─ Fix known issues (site association, iView)

2025 Q3-Q4: Features & Architecture
    │
    ├─ Multi-field support
    ├─ DICOM integration
    ├─ Microservices architecture
    └─ Enhanced UI/UX

2026+: Advanced ML & Clinical Integration
    │
    ├─ Active learning & explainability
    ├─ Treatment planning system plugins
    ├─ Regulatory compliance
    └─ Research publications
```

## Version Milestones

- **v1.0**: Fix critical issues, comprehensive docs, 80% test coverage
- **v1.5**: Multi-field support, DICOM integration, <100ms inference
- **v2.0**: Microservices architecture, uncertainty quantification, TPS integration
- **v3.0**: FDA clearance path, federated learning, clinical validation

## Feedback

This roadmap is a living document. Priorities may shift based on:
- Clinical feedback and requirements
- Technical discoveries and blockers
- Community contributions
- Regulatory requirements

Last updated: 2025-11-09
