# ChangeRisk

[![ChangeRisk CI](https://github.com/FernandoPCJ/changerisk/actions/workflows/ci.yml/badge.svg)](https://github.com/FernandoPCJ/changerisk/actions/workflows/ci.yml)

Machine Learning project for relative defect-risk prioritization of Pull Requests.

ChangeRisk analyzes structural characteristics of Pull Requests and produces a relative risk score that can be used to support review prioritization.

> **Important:** the `risk_score` is a relative ranking signal. It must not be interpreted as a calibrated probability that a Pull Request will introduce a defect.

---

## Live API

Production API:

https://changerisk.onrender.com

Swagger / OpenAPI:

https://changerisk.onrender.com/docs

Health check:

https://changerisk.onrender.com/health

The application is currently deployed on the Render Free tier. After periods of inactivity, the first request may experience a cold-start delay.

---

## Overview

ChangeRisk was developed as an end-to-end Machine Learning and MLOps project based on Pull Request data from the `pandas-dev/pandas` repository.

The project covers the complete lifecycle:

- GitHub data collection
- Pull Request and file-level enrichment
- defect-label construction with SZZ
- temporal target validation
- feature engineering
- data-leakage auditing
- exploratory data analysis
- baseline modeling
- model comparison
- ranking evaluation
- bootstrap uncertainty analysis
- operational threshold definition
- out-of-time evaluation
- temporal drift analysis
- model serialization
- inference layer
- REST API
- automated tests
- Docker
- MLflow experiment tracking
- Continuous Integration
- Continuous Deployment
- production deployment

---

## Problem

Traditional software-quality analysis often evaluates source code after changes have already been integrated.

ChangeRisk explores a complementary approach: using information available at Pull Request time to identify changes that deserve additional review attention.

The system is designed as a **risk-prioritization mechanism**, not as an automatic defect classifier.

The main operational question is:

> Which Pull Requests should receive higher review priority based on their relative estimated defect risk?

---

## Dataset

The final modeling population contains:

| Metric | Value |
|---|---:|
| Pull Requests | 6,822 |
| Positive cases | 114 |
| Negative cases | 6,708 |
| Positive rate | 1.67% |

The target variable is:

```text
observed_defect_90d

## Architecture

The ChangeRisk architecture covers the complete lifecycle from Pull Request data collection and SZZ-based target construction to model inference and production deployment.

![ChangeRisk Architecture and Delivery Flow](docs/images/changerisk-architecture-delivery-flow.png)